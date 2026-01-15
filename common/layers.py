"""
layers.py - TD-MPC2 layers with DEQ encoder support

Supports both standard MLP encoders and DEQ (Deep Equilibrium) encoders.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import from_modules
from copy import deepcopy
from implicit_layers.deq_encoder import DEQ_MLP, DEQ_CNN


class Ensemble(nn.Module):
    """Vectorized ensemble of modules."""

    def __init__(self, modules, **kwargs):
        super().__init__()
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
    """Random shift image augmentation."""

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
    """Normalizes pixel observations to [-0.5, 0.5]."""

    def forward(self, x):
        return x.div(255.).sub(0.5)


class SimNorm(nn.Module):
    """Simplicial normalization."""

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
    """Linear layer with LayerNorm, activation, and optionally dropout."""

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
    """MLP with LayerNorm, Mish activations, and optionally dropout."""
    if isinstance(mlp_dims, int):
        mlp_dims = [mlp_dims]
    dims = [in_dim] + mlp_dims + [out_dim]
    mlp_layers = nn.ModuleList()
    for i in range(len(dims) - 2):
        mlp_layers.append(NormedLinear(dims[i], dims[i + 1], dropout=dropout * (i == 0)))
    mlp_layers.append(NormedLinear(dims[-2], dims[-1], act=act) if act else nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*mlp_layers)


def conv(in_shape, num_channels, act=None):
    """Basic convolutional encoder for TD-MPC2."""
    assert in_shape[-1] == 64
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
    """Returns standard MLP/CNN encoders."""
    for k in cfg.obs_shape.keys():
        if k == 'state':
            out[k] = mlp(cfg.obs_shape[k][0] + cfg.task_dim, max(cfg.num_enc_layers - 1, 1) * [cfg.enc_dim],
                         cfg.latent_dim, act=SimNorm(cfg))
        elif k == 'rgb':
            out[k] = conv(cfg.obs_shape[k], cfg.num_channels, act=SimNorm(cfg))
        else:
            raise NotImplementedError(f"Encoder for observation type {k} not implemented.")
    return nn.ModuleDict(out)


def enc_deq(cfg, out={}):
    """Returns DEQ (Deep Equilibrium) encoders."""
    for k in cfg.obs_shape.keys():
        if k == 'state':
            out[k] = DEQ_MLP(
                input_dim=cfg.obs_shape[k][0] + cfg.task_dim,
                hidden_dim=cfg.enc_dim,
                output_dim=cfg.latent_dim,
                act=SimNorm(cfg),
                f_max_iter=getattr(cfg, 'deq_f_max_iter', 24),
                b_max_iter=getattr(cfg, 'deq_b_max_iter', 24),
                f_tol=getattr(cfg, 'deq_f_tol', 1e-3),
                b_tol=getattr(cfg, 'deq_b_tol', 1e-6),
                f_solver=getattr(cfg, 'deq_f_solver', 'anderson'),
                b_solver=getattr(cfg, 'deq_b_solver', 'anderson'),
                deq_num_layers=getattr(cfg, 'deq_num_layers', 2),
                dropout=getattr(cfg, 'deq_dropout', 0.0),
                log_stats=getattr(cfg, 'deq_log_stats', False),
                log_every_n_steps=getattr(cfg, 'deq_log_every_n_steps', 100),
                deq_lam=getattr(cfg, 'deq_lam', 0.5),
            )
        elif k == 'rgb':
            out[k] = DEQ_CNN(
                obs_shape=cfg.obs_shape[k],
                num_channels=cfg.num_channels,
                latent_dim=cfg.latent_dim,
                act=SimNorm(cfg),
                f_max_iter=getattr(cfg, 'deq_f_max_iter', 24),
                b_max_iter=getattr(cfg, 'deq_b_max_iter', 24),
                f_tol=getattr(cfg, 'deq_f_tol', 1e-3),
                b_tol=getattr(cfg, 'deq_b_tol', 1e-6),
                f_solver=getattr(cfg, 'deq_f_solver', 'anderson'),
                b_solver=getattr(cfg, 'deq_b_solver', 'anderson'),
                deq_num_layers=getattr(cfg, 'deq_num_layers', 2),
                log_stats=getattr(cfg, 'deq_log_stats', False),
                log_every_n_steps=getattr(cfg, 'deq_log_every_n_steps', 100),
                deq_lam=getattr(cfg, 'deq_lam', 0.5),
            )
        else:
            raise NotImplementedError(f"DEQ encoder for observation type {k} not implemented.")
    return nn.ModuleDict(out)


def api_model_conversion(target_state_dict, source_state_dict):
    """Converts checkpoint from old API to new torch.compile compatible API."""
    if "_detach_Qs_params.0.weight" in source_state_dict:
        return source_state_dict

    name_map = ['weight', 'bias', 'ln.weight', 'ln.bias']
    new_state_dict = dict()

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

    for prefix in ('_Qs.', '_detach_Qs_', '_target_Qs_'):
        for key in ('__batch_size', '__device'):
            new_key = prefix + 'params.' + key
            new_state_dict[new_key] = target_state_dict[new_key]

    for key in new_state_dict.keys():
        assert key in target_state_dict, f"key {key} not in target_state_dict"
    for key in target_state_dict.keys():
        if 'Qs' in key:
            assert key in new_state_dict, f"key {key} not in new_state_dict"
    for key in source_state_dict.keys():
        assert 'Qs' not in key, f"key {key} contains 'Qs'"

    new_state_dict['log_std_min'] = target_state_dict['log_std_min']
    new_state_dict['log_std_dif'] = target_state_dict['log_std_dif']
    if '_action_masks' in target_state_dict:
        new_state_dict['_action_masks'] = target_state_dict['_action_masks']

    source_state_dict.update(new_state_dict)
    return source_state_dict