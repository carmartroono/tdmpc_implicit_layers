import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import from_modules
from copy import deepcopy
from torchdeq import get_deq
from torchdeq.norm import apply_norm, reset_norm


class DEQStats:
    """Track DEQ convergence statistics for monitoring and debugging."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.forward_mean_iters = []
        self.forward_max_iters = []
        self.forward_mean_residuals = []
        self.forward_convergence_rates = []
        self.backward_mean_iters = []
        self.backward_max_iters = []
        self.backward_mean_residuals = []
        self.backward_convergence_rates = []
        self.singular_fallback_count = 0

    def update(self, info):
        """Update stats from DEQ solver info dict."""
        if 'nstep' in info:
            nstep = info['nstep']
            if torch.is_tensor(nstep):
                self.forward_mean_iters.append(nstep.mean().item())
                self.forward_max_iters.append(nstep.max().item())
            else:
                self.forward_mean_iters.append(nstep)
                self.forward_max_iters.append(nstep)
        if 'residual' in info:
            res = info['residual']
            if torch.is_tensor(res):
                self.forward_mean_residuals.append(res.mean().item())
            else:
                self.forward_mean_residuals.append(res)
        if 'converged' in info:
            conv = info['converged']
            if torch.is_tensor(conv):
                self.forward_convergence_rates.append(conv.float().mean().item())
            else:
                self.forward_convergence_rates.append(float(conv))

        # Backward pass stats (if available)
        if 'backward_nstep' in info:
            nstep = info['backward_nstep']
            if torch.is_tensor(nstep):
                self.backward_mean_iters.append(nstep.mean().item())
                self.backward_max_iters.append(nstep.max().item())
            else:
                self.backward_mean_iters.append(nstep)
                self.backward_max_iters.append(nstep)
        if 'backward_residual' in info:
            res = info['backward_residual']
            if torch.is_tensor(res):
                self.backward_mean_residuals.append(res.mean().item())
            else:
                self.backward_mean_residuals.append(res)
        if 'backward_converged' in info:
            conv = info['backward_converged']
            if torch.is_tensor(conv):
                self.backward_convergence_rates.append(conv.float().mean().item())
            else:
                self.backward_convergence_rates.append(float(conv))
        if 'singular_fallback' in info:
            self.singular_fallback_count += 1

    def get_summary(self):
        """Get summary statistics."""
        summary = {}
        if self.forward_mean_iters:
            summary['forward_iters_mean'] = sum(self.forward_mean_iters) / len(self.forward_mean_iters)
            summary['forward_iters_max'] = max(self.forward_max_iters)
        if self.forward_mean_residuals:
            summary['forward_residual_mean'] = sum(self.forward_mean_residuals) / len(self.forward_mean_residuals)
        summary['forward_convergence_rate'] = sum(self.forward_convergence_rates) / len(
            self.forward_convergence_rates) if self.forward_convergence_rates else 1.0

        if self.backward_mean_iters:
            summary['backward_iters_mean'] = sum(self.backward_mean_iters) / len(self.backward_mean_iters)
            summary['backward_iters_max'] = max(self.backward_max_iters)
        if self.backward_mean_residuals:
            summary['backward_residual_mean'] = sum(self.backward_mean_residuals) / len(self.backward_mean_residuals)
        summary['backward_convergence_rate'] = sum(self.backward_convergence_rates) / len(
            self.backward_convergence_rates) if self.backward_convergence_rates else 1.0

        summary['singular_fallback_rate'] = self.singular_fallback_count / max(len(self.forward_mean_iters), 1)

        return summary


class DEQFunc(nn.Module):
    """
    Improved DEQ function with configurable normalization.
    """

    def __init__(self, hidden_dim, num_layers=2, dropout=0.1, norm_type='layer'):
        """
        Args:
            hidden_dim: Hidden dimension size
            num_layers: Number of layers
            dropout: Dropout rate
            norm_type: Type of normalization - 'layer', 'spectral', or 'none'
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.norm_type = norm_type

        layers = []
        for i in range(num_layers):
            linear = nn.Linear(hidden_dim, hidden_dim)

            # Apply spectral normalization if specified
            if norm_type == 'spectral':
                linear = nn.utils.spectral_norm(linear)

            layers.append(linear)

            # Add layer normalization if specified
            if norm_type == 'layer':
                layers.append(nn.LayerNorm(hidden_dim))

            layers.append(nn.GELU())

            if dropout > 0 and i < num_layers - 1:
                layers.append(nn.Dropout(dropout))

        self.layers = nn.Sequential(*layers)

        # More conservative alpha
        self.alpha = nn.Parameter(torch.ones(1) * 0.01)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with smaller values for stability."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.05)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z):
        """Apply function with residual connection."""
        alpha_clamped = torch.clamp(self.alpha, min=0.001, max=0.05)
        return z + alpha_clamped * self.layers(z)


class DEQ_MLP(nn.Module):
    """
    Improved Deep Equilibrium MLP with configurable normalization.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, act=None,
                 f_max_iter=30, b_max_iter=30, f_tol=5e-3, b_tol=1e-5,
                 f_solver='broyden', b_solver='broyden',
                 deq_num_layers=2, dropout=0.05, norm_type='layer',
                 log_stats=True, log_every_n_steps=100, deq_lam=0.1, deq_tau=1.0):
        """
        Args:
            norm_type: Type of normalization for DEQ function
                      - 'layer': Use LayerNorm (default)
                      - 'spectral': Use Spectral Normalization
                      - 'none': No normalization
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.act = act if act is not None else nn.GELU()
        self.log_stats = log_stats
        self.log_every_n_steps = log_every_n_steps
        self.norm_type = norm_type

        self.f_max_iter = f_max_iter
        self.f_tol = f_tol
        self.deq_tau = deq_tau
        self.deq_lam = deq_lam

        # Warm-up period
        self.warmup_steps = 1000
        self.current_step = 0

        # Statistics tracking
        self.stats = DEQStats()
        self.step_counter = 0

        # Input projection (always with LayerNorm for stability)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            self.act
        )

        # Improved DEQ function with configurable normalization
        self.deq_func = DEQFunc(
            hidden_dim=hidden_dim,
            num_layers=deq_num_layers,
            dropout=dropout,
            norm_type=norm_type
        )

        # Output projection (always with LayerNorm for stability)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

        # Skip connection with learnable weight
        self.skip_proj = nn.Linear(input_dim, output_dim)
        self.skip_weight = nn.Parameter(torch.ones(1) * 0.5)

        # Create DEQ solver
        self.deq = get_deq(
            f_max_iter=f_max_iter,
            b_max_iter=b_max_iter,
            f_tol=f_tol,
            b_tol=b_tol,
            f_solver=f_solver,
            b_solver=b_solver,
            f_lam=deq_lam,
            b_lam=deq_lam,
            f_tau=deq_tau,
            b_tau=deq_tau
        )

    @torch._dynamo.disable
    def forward(self, x):
        """
        Forward pass with adaptive solver.
        """
        self.current_step += 1

        # Project input
        z0 = self.input_proj(x)

        # Apply reset_norm only if using LayerNorm and it's available
        if self.norm_type == 'layer' and hasattr(self.act, 'reset_parameters'):
            reset_norm(self.deq_func)

        # Improved fallback with relaxation
        try:
            z_star, info = self.deq(self.deq_func, z0)
        except RuntimeError as e:
            if 'singular' in str(e).lower():
                if self.training:
                    print(f"[Step {self.current_step}] Singular matrix, using fallback")

                z_star = z0.clone()
                # Adaptive relaxation
                omega = 0.1 if self.current_step < self.warmup_steps else 0.3

                for i in range(self.f_max_iter):
                    z_new = self.deq_func(z_star)
                    z_star = (1 - omega) * z_star + omega * z_new

                    # Early stopping
                    if i % 5 == 0:
                        diff = z_star - z_new
                        residual = torch.norm(diff, dim=-1).mean().item()

                        tolerance_scale = 5.0 if self.current_step < self.warmup_steps else 1.0
                        if residual < self.f_tol * tolerance_scale:
                            break

                diff = z_star - self.deq_func(z_star)
                residual = torch.norm(diff, dim=-1).mean().item()
                converged = residual < self.f_tol * 5
                info = {
                    'nstep': i + 1,
                    'residual': residual,
                    'converged': converged,
                    'singular_fallback': True
                }
            else:
                raise e

        # Log statistics
        if self.log_stats and self.training:
            self.stats.update(info)
            self.step_counter += 1

            if self.step_counter % self.log_every_n_steps == 0:
                summary = self.stats.get_summary()
                print(f"[DEQ Stats @ {self.step_counter}] "
                      f"Norm: {self.norm_type}, "
                      f"Iters: {summary.get('forward_iters_mean', 0):.1f}, "
                      f"Residual: {summary.get('forward_residual_mean', 0):.2e}, "
                      f"Conv Rate: {summary.get('forward_convergence_rate', 0):.2%}")
                self.stats.reset()

        # Handle trajectory outputs
        if isinstance(z_star, (list, tuple)):
            z_final = z_star[-1]
        else:
            z_final = z_star

        # Apply norm only if using LayerNorm and it's available
        if self.norm_type == 'layer' and hasattr(self.act, 'reset_parameters'):
            apply_norm(self.deq_func, z_final - z0, self.training)

        # Project to output
        out = self.output_proj(z_final)

        # Adaptive skip connection
        skip = self.skip_proj(x)
        skip_w = torch.sigmoid(self.skip_weight)
        out = (1 - skip_w) * out + skip_w * skip

        return out


class DEQConvFunc(nn.Module):
    """
    Improved DEQ convolutional function with spectral normalization.
    """

    def __init__(self, num_channels, num_layers=2, norm_type='batch'):
        super().__init__()
        self.num_channels = num_channels
        self.num_layers = num_layers
        self.norm_type = norm_type

        layers = []
        for i in range(num_layers):
            conv = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
            conv = nn.utils.spectral_norm(conv)
            layers.append(conv)

            if norm_type == 'batch':
                layers.append(nn.BatchNorm2d(num_channels))
            elif norm_type == 'instance':
                layers.append(nn.InstanceNorm2d(num_channels))
            elif norm_type == 'none':
                pass
            else:
                raise ValueError(f"Unknown norm_type: {norm_type}. Use 'batch', 'instance', or 'none'.")

            layers.append(nn.GELU())

        self.layers = nn.Sequential(*layers)

        self.alpha = nn.Parameter(torch.ones(1) * 0.01)  #

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Ð˜Ð—ÐœÐ•ÐÐ•ÐÐ˜Ð•: ÐœÐµÐ½ÑŒÑˆÐ¸Ð¹ gain
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu', a=0.05)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z):
        alpha_clamped = torch.clamp(self.alpha, min=0.001, max=0.05)
        return z + alpha_clamped * self.layers(z)


class DEQ_CNN(nn.Module):
    """
    Improved Deep Equilibrium CNN with better stability.
    """

    def __init__(self, obs_shape, num_channels, latent_dim, act=None,
                 f_max_iter=30, b_max_iter=30, f_tol=5e-3, b_tol=1e-5,
                 f_solver='broyden', b_solver='broyden',
                 deq_num_layers=2, norm_type='batch',
                 log_stats=True, log_every_n_steps=100, deq_lam=0.1, deq_tau=1.0):
        super().__init__()
        self.obs_shape = obs_shape
        self.num_channels = num_channels
        self.latent_dim = latent_dim
        self.act = act if act is not None else nn.GELU()
        self.log_stats = log_stats
        self.log_every_n_steps = log_every_n_steps
        self.norm_type = norm_type

        self.f_max_iter = f_max_iter
        self.f_tol = f_tol
        self.deq_lam = deq_lam
        self.deq_tau = deq_tau

        # ÐÐžÐ’ÐžÐ•: Warm-up
        self.warmup_steps = 1000
        self.current_step = 0

        self.stats = DEQStats()
        self.step_counter = 0

        def get_norm_layer(channels):
            if norm_type == 'batch':
                return nn.BatchNorm2d(channels)
            elif norm_type == 'instance':
                return nn.InstanceNorm2d(channels)
            elif norm_type == 'none':
                return nn.Identity()
            else:
                raise ValueError(f"Unknown norm_type: {norm_type}")

        # Input convolution
        self.input_conv = nn.Sequential(
            nn.Conv2d(obs_shape[0], num_channels, kernel_size=7, stride=2),
            get_norm_layer(num_channels),
            self.act
        )

        # DEQ function
        self.deq_func = DEQConvFunc(
            num_channels=num_channels,
            num_layers=deq_num_layers,
            norm_type=norm_type
        )

        # Downsampling
        self.downsample1 = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=5, stride=2),
            get_norm_layer(num_channels),
            self.act,
        )
        self.downsample2 = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=2),
            get_norm_layer(num_channels),
            self.act,
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1),
            get_norm_layer(num_channels),
            self.act,
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # Final projection
        flattened_size = num_channels * 4 * 4
        self.output_proj = nn.Sequential(
            nn.Linear(flattened_size, latent_dim),
            nn.LayerNorm(latent_dim)
        )

        # DEQ solver
        self.deq = get_deq(
            f_max_iter=f_max_iter,
            b_max_iter=b_max_iter,
            f_tol=f_tol,
            b_tol=b_tol,
            f_solver=f_solver,
            b_solver=b_solver,
            f_lam=deq_lam,
            b_lam=deq_lam,
            f_tau=deq_tau,
            b_tau=deq_tau
        )

    @torch._dynamo.disable
    def forward(self, x):
        """Forward pass with noise injection during warm-up."""
        self.current_step += 1

        # Input convolution
        z0 = self.input_conv(x)

        if hasattr(self.act, 'reset_parameters'):
            reset_norm(self.deq_func)

        # Solve with improved fallback
        try:
            z_star, info = self.deq(self.deq_func, z0)
        except RuntimeError as e:
            if 'singular' in str(e).lower():
                if self.training:
                    print(f"[Step {self.current_step}] CNN Singular matrix, using fallback")

                z_star = z0.clone()
                omega = 0.1 if self.current_step < self.warmup_steps else 0.3

                for i in range(self.f_max_iter):
                    z_new = self.deq_func(z_star)
                    z_star = (1 - omega) * z_star + omega * z_new

                    if i % 5 == 0:
                        diff = z_star - z_new
                        residual = torch.norm(diff, dim=(-3, -2, -1)).mean().item()
                        tolerance_scale = 5.0 if self.current_step < self.warmup_steps else 1.0
                        if residual < self.f_tol * tolerance_scale:
                            break

                diff = z_star - self.deq_func(z_star)
                residual = torch.norm(diff, dim=(-3, -2, -1)).mean().item()
                converged = residual < self.f_tol * 5
                info = {
                    'nstep': i + 1,
                    'residual': residual,
                    'converged': converged,
                    'singular_fallback': True
                }
            else:
                raise e

        if self.log_stats and self.training:
            self.stats.update(info)
            self.step_counter += 1

            if self.step_counter % self.log_every_n_steps == 0:
                summary = self.stats.get_summary()
                print(f"[DEQ CNN Stats @ {self.step_counter}] "
                      f"Iters: {summary.get('forward_iters_mean', 0):.1f}, "
                      f"Residual: {summary.get('forward_residual_mean', 0):.2e}")
                self.stats.reset()

        if isinstance(z_star, (list, tuple)):
            z_final = z_star[-1]
        else:
            z_final = z_star

        if hasattr(self.act, 'reset_parameters'):
            apply_norm(self.deq_func, z_final - z0, self.training)

        # Downsampling and projection
        z_final = self.downsample1(z_final)
        z_final = self.downsample2(z_final)
        z_final = self.final_conv(z_final)
        z_final = self.adaptive_pool(z_final)

        out = z_final.flatten(1)
        out = self.output_proj(out)

        return out