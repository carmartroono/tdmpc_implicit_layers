import torch
import torch.nn as nn
from torchdeq import get_deq


class DEQStats:
    """Track DEQ convergence statistics."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.forward_iters = []
        self.forward_residuals = []

    def update(self, info):
        if 'nstep' in info:
            nstep = info['nstep']
            self.forward_iters.append(nstep.mean().item() if torch.is_tensor(nstep) else nstep)
        if 'rel_lowest' in info:
            res = info['rel_lowest']
            self.forward_residuals.append(res.mean().item() if torch.is_tensor(res) else res)

    def get_summary(self):
        summary = {}
        if self.forward_iters:
            summary['iters_mean'] = sum(self.forward_iters) / len(self.forward_iters)
        if self.forward_residuals:
            summary['residual_mean'] = sum(self.forward_residuals) / len(self.forward_residuals)
        return summary


class DEQFunc(nn.Module):
    """DEQ implicit function with spectral normalization for contraction."""

    def __init__(self, hidden_dim, num_layers=2, dropout=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim

        layers = []
        for i in range(num_layers):
            layers.append(nn.utils.spectral_norm(nn.Linear(hidden_dim, hidden_dim)))
            layers.append(nn.GELU())
            if dropout > 0 and i < num_layers - 1:
                layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z, x_inj):
        return self.net(z) + x_inj


class DEQ_MLP(nn.Module):
    """Deep Equilibrium MLP encoder."""

    def __init__(self, input_dim, hidden_dim, output_dim, act=None,
                 f_max_iter=24, b_max_iter=24, f_tol=1e-3, b_tol=1e-6,
                 f_solver='anderson', b_solver='anderson',
                 deq_num_layers=2, dropout=0.0,
                 log_stats=False, log_every_n_steps=100,
                 deq_lam=0.5, **kwargs):
        super().__init__()
        self.act = act
        self.log_stats = log_stats
        self.log_every_n_steps = log_every_n_steps

        self.stats = DEQStats()
        self.step_counter = 0

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU()
        )

        self.deq_func = DEQFunc(
            hidden_dim=hidden_dim,
            num_layers=deq_num_layers,
            dropout=dropout
        )

        self.output_proj = nn.Linear(hidden_dim, output_dim)

        self.deq = get_deq(
            f_max_iter=f_max_iter,
            b_max_iter=b_max_iter,
            f_tol=f_tol,
            b_tol=b_tol,
            f_solver=f_solver,
            b_solver=b_solver,
            f_lam=deq_lam,
            b_lam=deq_lam,
        )

    def forward(self, x):
        x_proj = self.input_proj(x)
        z0 = x_proj

        func = lambda z: self.deq_func(z, x_proj)
        z_star, info = self.deq(func, z0)

        if self.log_stats and self.training:
            self.stats.update(info)
            self.step_counter += 1
            if self.step_counter % self.log_every_n_steps == 0:
                summary = self.stats.get_summary()
                print(f"[DEQ @ {self.step_counter}] Iters: {summary.get('iters_mean', 0):.1f}, Res: {summary.get('residual_mean', 0):.2e}")
                self.stats.reset()

        z_final = z_star[-1] if isinstance(z_star, (list, tuple)) else z_star
        out = self.output_proj(z_final)

        if self.act is not None:
            out = self.act(out)

        return out


class DEQConvFunc(nn.Module):
    """DEQ convolutional function with spectral normalization."""

    def __init__(self, num_channels, num_layers=2):
        super().__init__()

        layers = []
        for _ in range(num_layers):
            layers.append(nn.utils.spectral_norm(nn.Conv2d(num_channels, num_channels, 3, padding=1)))
            layers.append(nn.GELU())

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                with torch.no_grad():
                    m.weight.mul_(0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z, x_inj):
        return self.net(z) + x_inj


class DEQ_CNN(nn.Module):
    """Deep Equilibrium CNN encoder for images."""

    def __init__(self, obs_shape, num_channels, latent_dim, act=None,
                 f_max_iter=24, b_max_iter=24, f_tol=1e-3, b_tol=1e-6,
                 f_solver='anderson', b_solver='anderson',
                 deq_num_layers=2,
                 log_stats=False, log_every_n_steps=100,
                 deq_lam=0.5, **kwargs):
        super().__init__()
        self.act = act
        self.log_stats = log_stats
        self.log_every_n_steps = log_every_n_steps

        self.stats = DEQStats()
        self.step_counter = 0

        self.input_conv = nn.Sequential(
            nn.Conv2d(obs_shape[0], num_channels, 7, stride=2),
            nn.GELU()
        )

        self.deq_func = DEQConvFunc(num_channels, deq_num_layers)

        self.downsample = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, 5, stride=2), nn.GELU(),
            nn.Conv2d(num_channels, num_channels, 3, stride=2), nn.GELU(),
            nn.Conv2d(num_channels, num_channels, 3, stride=1), nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten()
        )

        self.output_proj = nn.Linear(num_channels * 16, latent_dim)

        self.deq = get_deq(
            f_max_iter=f_max_iter,
            b_max_iter=b_max_iter,
            f_tol=f_tol,
            b_tol=b_tol,
            f_solver=f_solver,
            b_solver=b_solver,
            f_lam=deq_lam,
            b_lam=deq_lam,
        )

    def forward(self, x):
        x_proj = self.input_conv(x)
        z0 = x_proj

        func = lambda z: self.deq_func(z, x_proj)
        z_star, info = self.deq(func, z0)

        if self.log_stats and self.training:
            self.stats.update(info)
            self.step_counter += 1
            if self.step_counter % self.log_every_n_steps == 0:
                summary = self.stats.get_summary()
                print(f"[DEQ CNN @ {self.step_counter}] Iters: {summary.get('iters_mean', 0):.1f}")
                self.stats.reset()

        z_final = z_star[-1] if isinstance(z_star, (list, tuple)) else z_star
        out = self.downsample(z_final)
        out = self.output_proj(out)

        if self.act is not None:
            out = self.act(out)

        return out