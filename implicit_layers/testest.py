import torch
import torch.nn as nn
from torchdeq import DEQ  # https://github.com/locuslab/deq

# ------------------------------
# DEQ MLP для вектора состояния
# ------------------------------
class DEQ_MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, act=nn.ReLU):
        super().__init__()
        self.f_theta = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            act(),
            nn.Linear(hidden_dim, hidden_dim),
            act(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.deq_layer = DEQ(f=self.f_theta, solver='broyden', max_iter=25, tol=1e-5)

    def forward(self, x):
        # DEQ forward: решаем z* = f_theta(z*, x)
        z_star = self.deq_layer(x)
        return z_star


# ------------------------------
# DEQ CNN для изображений (rgb)
# ------------------------------
class DEQ_CNN(nn.Module):
    def __init__(self, input_shape, num_channels, latent_dim, act=nn.ReLU):
        super().__init__()
        C, H, W = input_shape
        self.conv_f = nn.Sequential(
            nn.Conv2d(C, num_channels, kernel_size=3, stride=2, padding=1),
            act(),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=2, padding=1),
            act(),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=2, padding=1),
            act(),
            nn.Flatten(),
            nn.Linear((H//8)*(W//8)*num_channels, latent_dim)
        )
        self.deq_layer = DEQ(f=self.conv_f, solver='broyden', max_iter=25, tol=1e-5)

    def forward(self, x):
        # DEQ forward для CNN
        z_star = self.deq_layer(x)
        return z_star


# ------------------------------
# Интеграция в enc()
# ------------------------------
def enc(cfg, out={}):
    for k in cfg.obs_shape.keys():
        if k == 'state':
            out[k] = DEQ_MLP(
                input_dim=cfg.obs_shape[k][0] + cfg.task_dim,
                hidden_dim=max(cfg.num_enc_layers-1, 1)*cfg.enc_dim,
                latent_dim=cfg.latent_dim,
                act=nn.ReLU  # или SimNorm(cfg)
            )
        elif k == 'rgb':
            out[k] = DEQ_CNN(
                input_shape=cfg.obs_shape[k],
                num_channels=cfg.num_channels,
                latent_dim=cfg.latent_dim,
                act=nn.ReLU  # или SimNorm(cfg)
            )
        else:
            raise NotImplementedError(f"Encoder for observation type {k} not implemented.")
    return nn.ModuleDict(out)


# ------------------------------
# Forward example
# ------------------------------
# obs = {"state": s, "rgb": img}
encoders = enc(cfg)
z_state = encoders["state"](obs["state"])
z_rgb   = encoders["rgb"](obs["rgb"])
z = torch.cat([z_state, z_rgb], dim=-1)  # общий latent для dynamics