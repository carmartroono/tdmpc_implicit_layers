import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import from_modules
from copy import deepcopy
from torchdeq import get_deq
from torchdeq.norm import apply_norm, reset_norm

class Ensemble(nn.Module):
	"""
	Vectorized ensemble of modules.
	"""

	def __init__(self, modules, **kwargs):
		super().__init__()
		# combine_state_for_ensemble causes graph breaks
		self.params = from_modules(*modules, as_module=True)
		with self.params[0].data.to("meta").to_module(modules[0]):
			self.module = deepcopy(modules[0])
		self._repr = str(modules[0])
		self._n = len(modules)

	def __len__(self):
		return self._n

	def _call(self, params, *args, **kwargs):
		with params.to_module(self.module):
			return self.module(*args, **kwargs)

	def forward(self, *args, **kwargs):
		return torch.vmap(self._call, (0, None), randomness="different")(self.params, *args, **kwargs)

	def __repr__(self):
		return f'Vectorized {len(self)}x ' + self._repr


class ShiftAug(nn.Module):
	"""
	Random shift image augmentation.
	Adapted from https://github.com/facebookresearch/drqv2
	"""
	def __init__(self, pad=3):
		super().__init__()
		self.pad = pad
		self.padding = tuple([self.pad] * 4)

	def forward(self, x):
		x = x.float()
		n, _, h, w = x.size()
		assert h == w
		x = F.pad(x, self.padding, 'replicate')
		eps = 1.0 / (h + 2 * self.pad)
		arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
		arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
		base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
		base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
		shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
		shift *= 2.0 / (h + 2 * self.pad)
		grid = base_grid + shift
		return F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)


class PixelPreprocess(nn.Module):
	"""
	Normalizes pixel observations to [-0.5, 0.5].
	"""

	def __init__(self):
		super().__init__()

	def forward(self, x):
		return x.div(255.).sub(0.5)


class SimNorm(nn.Module):
	"""
	Simplicial normalization.
	Adapted from https://arxiv.org/abs/2204.00616.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.dim = cfg.simnorm_dim

	def forward(self, x):
		shp = x.shape
		x = x.view(*shp[:-1], -1, self.dim)
		x = F.softmax(x, dim=-1)
		return x.view(*shp)

	def __repr__(self):
		return f"SimNorm(dim={self.dim})"


class NormedLinear(nn.Linear):
	"""
	Linear layer with LayerNorm, activation, and optionally dropout.
	"""

	def __init__(self, *args, dropout=0., act=None, **kwargs):
		super().__init__(*args, **kwargs)
		self.ln = nn.LayerNorm(self.out_features)
		if act is None:
			act = nn.Mish(inplace=False)
		self.act = act
		self.dropout = nn.Dropout(dropout, inplace=False) if dropout else None

	def forward(self, x):
		x = super().forward(x)
		if self.dropout:
			x = self.dropout(x)
		return self.act(self.ln(x))

	def __repr__(self):
		repr_dropout = f", dropout={self.dropout.p}" if self.dropout else ""
		return f"NormedLinear(in_features={self.in_features}, "\
			f"out_features={self.out_features}, "\
			f"bias={self.bias is not None}{repr_dropout}, "\
			f"act={self.act.__class__.__name__})"


def mlp(in_dim, mlp_dims, out_dim, act=None, dropout=0.):
	"""
	Basic building block of TD-MPC2.
	MLP with LayerNorm, Mish activations, and optionally dropout.
	"""
	if isinstance(mlp_dims, int):
		mlp_dims = [mlp_dims]
	dims = [in_dim] + mlp_dims + [out_dim]
	mlp = nn.ModuleList()
	for i in range(len(dims) - 2):
		mlp.append(NormedLinear(dims[i], dims[i+1], dropout=dropout*(i==0)))
	mlp.append(NormedLinear(dims[-2], dims[-1], act=act) if act else nn.Linear(dims[-2], dims[-1]))
	return nn.Sequential(*mlp)


def conv(in_shape, num_channels, act=None):
	"""
	Basic convolutional encoder for TD-MPC2 with raw image observations.
	4 layers of convolution with ReLU activations, followed by a linear layer.
	"""
	assert in_shape[-1] == 64 # assumes rgb observations to be 64x64
	layers = [
		ShiftAug(), PixelPreprocess(),
		nn.Conv2d(in_shape[0], num_channels, 7, stride=2), nn.ReLU(inplace=False),
		nn.Conv2d(num_channels, num_channels, 5, stride=2), nn.ReLU(inplace=False),
		nn.Conv2d(num_channels, num_channels, 3, stride=2), nn.ReLU(inplace=False),
		nn.Conv2d(num_channels, num_channels, 3, stride=1), nn.Flatten()]
	if act:
		layers.append(act)
	return nn.Sequential(*layers)


def enc(cfg, out={}):
	"""
	Returns a dictionary of encoders for each observation in the dict.
	"""
	for k in cfg.obs_shape.keys():
		if k == 'state':
			out[k] = mlp(cfg.obs_shape[k][0] + cfg.task_dim, max(cfg.num_enc_layers-1, 1)*[cfg.enc_dim], cfg.latent_dim, act=SimNorm(cfg))
		elif k == 'rgb':
			out[k] = conv(cfg.obs_shape[k], cfg.num_channels, act=SimNorm(cfg))
		else:
			raise NotImplementedError(f"Encoder for observation type {k} not implemented.")
	return nn.ModuleDict(out)


def api_model_conversion(target_state_dict, source_state_dict):
	"""
	Converts a checkpoint from our old API to the new torch.compile compatible API.
	"""
	# check whether checkpoint is already in the new format
	if "_detach_Qs_params.0.weight" in source_state_dict:
		return source_state_dict

	name_map = ['weight', 'bias', 'ln.weight', 'ln.bias']
	new_state_dict = dict()

	# rename keys
	for key, val in list(source_state_dict.items()):
		if key.startswith('_Qs.'):
			num = key[len('_Qs.params.'):]
			new_key = str(int(num) // 4) + "." + name_map[int(num) % 4]
			new_total_key = "_Qs.params." + new_key
			del source_state_dict[key]
			new_state_dict[new_total_key] = val
			new_total_key = "_detach_Qs_params." + new_key
			new_state_dict[new_total_key] = val
		elif key.startswith('_target_Qs.'):
			num = key[len('_target_Qs.params.'):]
			new_key = str(int(num) // 4) + "." + name_map[int(num) % 4]
			new_total_key = "_target_Qs_params." + new_key
			del source_state_dict[key]
			new_state_dict[new_total_key] = val

	# add batch_size and device from target_state_dict to new_state_dict
	for prefix in ('_Qs.', '_detach_Qs_', '_target_Qs_'):
		for key in ('__batch_size', '__device'):
			new_key = prefix + 'params.' + key
			new_state_dict[new_key] = target_state_dict[new_key]

	# check that every key in new_state_dict is in target_state_dict
	for key in new_state_dict.keys():
		assert key in target_state_dict, f"key {key} not in target_state_dict"
	# check that all Qs keys in target_state_dict are in new_state_dict
	for key in target_state_dict.keys():
		if 'Qs' in key:
			assert key in new_state_dict, f"key {key} not in new_state_dict"
	# check that source_state_dict contains no Qs keys
	for key in source_state_dict.keys():
		assert 'Qs' not in key, f"key {key} contains 'Qs'"

	# copy log_std_min and log_std_max from target_state_dict to new_state_dict
	new_state_dict['log_std_min'] = target_state_dict['log_std_min']
	new_state_dict['log_std_dif'] = target_state_dict['log_std_dif']
	if '_action_masks' in target_state_dict:
		new_state_dict['_action_masks'] = target_state_dict['_action_masks']

	# copy new_state_dict to source_state_dict
	source_state_dict.update(new_state_dict)

	return source_state_dict


class DEQ_MLP(nn.Module):
    """
    Deep Equilibrium MLP encoder for state observations.
    Replaces the explicit MLP with an implicit DEQ layer.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, act=None, f_max_iter=40,
                 b_max_iter=40, f_tol=1e-3, b_tol=1e-6, f_solver='anderson',
                 b_solver='anderson'):
        """
        Args:
            input_dim: Input dimension (obs_shape[k][0] + task_dim)
            hidden_dim: Hidden dimension for DEQ layers (cfg.enc_dim)
            output_dim: Output dimension (cfg.latent_dim)
            act: Activation function (SimNorm in original code)
            f_max_iter: Maximum iterations for forward solver
            b_max_iter: Maximum iterations for backward solver
            f_tol: Tolerance for forward solver
            b_tol: Tolerance for backward solver
            f_solver: Forward solver type ('anderson', 'broyden', 'fixed_point')
            b_solver: Backward solver type
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.act = act if act is not None else nn.ReLU()

        # Input projection: projects input to hidden dimension
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # DEQ function: defines the implicit layer f(z) = z
        # This replaces the explicit MLP layers
        self.deq_func = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            self.act,
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Output projection: projects from hidden to output dimension
        self.output_proj = nn.Linear(hidden_dim, output_dim)

        # Create DEQ solver
        self.deq = get_deq(
            f_max_iter=f_max_iter,
            b_max_iter=b_max_iter,
            f_tol=f_tol,
            b_tol=b_tol,
            f_solver=f_solver,
            b_solver=b_solver
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # Project input to hidden dimension
        z0 = self.input_proj(x)

        # Apply normalization if using SimNorm
        if hasattr(self.act, 'reset_parameters'):
            reset_norm(self.deq_func)

        # Solve for equilibrium: find z* such that f(z*) = z*
        # This is the key difference from explicit layers
        z_star, info = self.deq(self.deq_func, z0)
		#z_star = z_star[-1]
        # Apply normalization after DEQ
        if hasattr(self.act, 'reset_parameters'):
            apply_norm(self.deq_func, z_star[-1] - z0, self.training)

        # Project to output dimension
        out = self.output_proj(z_star[-1])

        return out


class DEQ_CNN(nn.Module):
    """
    Deep Equilibrium CNN encoder for RGB observations.
    Replaces the explicit CNN with an implicit DEQ layer.
    Updated to better match original output size (~num_channels * 16).
    """

    def __init__(self, obs_shape, num_channels, latent_dim, act=None, f_max_iter=40,
                 b_max_iter=40, f_tol=1e-3, b_tol=1e-6, f_solver='anderson',
                 b_solver='anderson'):
        """
        Args:
            obs_shape: Shape of RGB observation (C, H, W)
            num_channels: Number of channels in conv layers (cfg.num_channels)
            latent_dim: Output dimension (cfg.latent_dim, to match original flattened size)
            act: Activation function (SimNorm in original code)
            f_max_iter: Maximum iterations for forward solver
            b_max_iter: Maximum iterations for backward solver
            f_tol: Tolerance for forward solver
            b_tol: Tolerance for backward solver
            f_solver: Forward solver type
            b_solver: Backward solver type
        """
        super().__init__()
        self.obs_shape = obs_shape
        self.num_channels = num_channels
        self.latent_dim = latent_dim
        self.act = act if act is not None else nn.ReLU()

        # Input convolution: matches original first conv
        self.input_conv = nn.Conv2d(
            obs_shape[0],
            num_channels,
            kernel_size=7,
            stride=2
        )  # No pad, size ~29x29 for 64x64

        # DEQ function: implicit convolutional layer
        self.deq_func = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
            self.act,
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
        )

        # Additional downsamplings to match original (~4x4 feature map)
        self.downsample1 = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=5, stride=2),  # ~13x13
            self.act,
        )
        self.downsample2 = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=2),  # ~6x6
            self.act,
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1),  # ~4x4
            self.act,
        )

        # Final activation (SimNorm on flattened, as in original)
        self.final_act = self.act  # Reuse act (SimNorm)

        # Projection if flattened size != latent_dim
        flattened_size = num_channels * 4 * 4  # Assuming ~4x4 after convs
        self.output_proj = nn.Linear(flattened_size, latent_dim) if flattened_size != latent_dim else None

        # Create DEQ solver
        self.deq = get_deq(
            f_max_iter=f_max_iter,
            b_max_iter=b_max_iter,
            f_tol=f_tol,
            b_tol=b_tol,
            f_solver=f_solver,
            b_solver=b_solver
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, C, H, W)
        Returns:
            Output tensor of shape (batch_size, latent_dim)
        """
        # Input convolution
        z0 = self.input_conv(x)

        # Apply normalization if applicable
        if hasattr(self.act, 'reset_parameters'):
            reset_norm(self.deq_func)

        # Solve for equilibrium
        z_star, info = self.deq(self.deq_func, z0)

        # Apply normalization after DEQ
        if hasattr(self.act, 'reset_parameters'):
            apply_norm(self.deq_func, z_star - z0, self.training)

        # Additional downsamplings
        z_star = self.downsample1(z_star)
        z_star = self.downsample2(z_star)
        z_star = self.final_conv(z_star)

        # Flatten
        out = z_star.flatten(1)

        # Apply final SimNorm on flattened
        out = self.final_act(out)

        # Project if needed
        if self.output_proj is not None:
            out = self.output_proj(out)

        return out
def enc_deq(cfg, out={}):
    """
    Returns a dictionary of DEQ encoders for each observation in the dict.
    This replaces the original enc() function.
    """
    for k in cfg.obs_shape.keys():
        if k == 'state':
            out[k] = DEQ_MLP(
                input_dim=cfg.obs_shape[k][0] + cfg.task_dim,
                hidden_dim=cfg.enc_dim,
                output_dim=cfg.latent_dim,
                act=SimNorm(cfg),
                f_max_iter=getattr(cfg, 'deq_f_max_iter', 40),
                b_max_iter=getattr(cfg, 'deq_b_max_iter', 40),
                f_tol=getattr(cfg, 'deq_f_tol', 1e-3),
                b_tol=getattr(cfg, 'deq_b_tol', 1e-6),
                f_solver=getattr(cfg, 'deq_f_solver', 'anderson'),
                b_solver=getattr(cfg, 'deq_b_solver', 'anderson'),
            )
        elif k == 'rgb':
            out[k] = DEQ_CNN(
                obs_shape=cfg.obs_shape[k],
                num_channels=cfg.num_channels,
                latent_dim=cfg.latent_dim,  # Добавлено для матча размеров
                act=SimNorm(cfg),
                f_max_iter=getattr(cfg, 'deq_f_max_iter', 40),
                b_max_iter=getattr(cfg, 'deq_b_max_iter', 40),
                f_tol=getattr(cfg, 'deq_f_tol', 1e-3),
                b_tol=getattr(cfg, 'deq_b_tol', 1e-6),
                f_solver=getattr(cfg, 'deq_f_solver', 'anderson'),
                b_solver=getattr(cfg, 'deq_b_solver', 'anderson'),
            )
        else:
            raise NotImplementedError(f"DEQ encoder for observation type {k} not implemented.")
    return nn.ModuleDict(out)