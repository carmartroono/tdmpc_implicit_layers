import torch
import torch.nn.functional as F
from torch import nn
import numpy as np


class _System_contractive(nn.Module):
    """Contractive system core guaranteeing stability through matrix parameterization."""

    def __init__(self, nx, ny, nu, nq, sigma, epsilon, device, bias=False, alpha=0.0, linear_output=False):
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.nu = nu
        self.nq = nq
        self.epsilon = epsilon
        self.device = device
        self.alpha = alpha

        std = 0.1 / np.sqrt(nx)

        self.Pstar = nn.Parameter(torch.randn(nx, nx, device=device) * std)
        self.Chi = nn.Parameter(torch.randn(nx, nq, device=device) * std)
        self.Y1 = nn.Parameter(torch.randn(nx, nx, device=device) * std)
        self.B2 = nn.Parameter(torch.randn(nx, nu, device=device) * std)
        self.D12 = nn.Parameter(torch.randn(nq, nu, device=device) * std)
        self.C2 = nn.Parameter(torch.randn(ny, nx, device=device) * std)

        if linear_output:
            self.register_buffer('D21', torch.zeros(ny, nq, device=device))
        else:
            self.D21 = nn.Parameter(torch.randn(ny, nq, device=device) * std)
        self.D22 = nn.Parameter(torch.randn(ny, nu, device=device) * std)

        if bias:
            self.bx = nn.Parameter(torch.zeros(nx, 1, device=device))
            self.bv = nn.Parameter(torch.zeros(nq, 1, device=device))
            self.by = nn.Parameter(torch.zeros(ny, 1, device=device))
        else:
            self.register_buffer('bx', torch.zeros(nx, 1, device=device))
            self.register_buffer('bv', torch.zeros(nq, 1, device=device))
            self.register_buffer('by', torch.zeros(ny, 1, device=device))

        self.X = nn.Parameter(torch.randn(nx + nq, nx + nq, device=device) * std)

        activations = {'tanh': nn.Tanh(), 'sigmoid': nn.Sigmoid(), 'relu': nn.ReLU(), 'identity': nn.Identity()}
        self.act = activations.get(sigma, nn.Tanh())

    def _compute_constrained_matrices(self):
        P = 0.5 * F.linear(self.Pstar, self.Pstar) + self.epsilon * torch.eye(self.nx, device=self.device)
        H = F.linear(self.X, self.X) + self.epsilon * torch.eye(self.nx + self.nq, device=self.device)

        h1, h2 = torch.split(H, (self.nx, self.nq), dim=0)
        H1, H2 = torch.split(h1, (self.nx, self.nq), dim=1)
        _, H4 = torch.split(h2, (self.nx, self.nq), dim=1)

        Y = -0.5 * (H1 + self.alpha * P + self.Y1 - self.Y1.T)
        Lambda = 0.5 * torch.diag_embed(torch.diagonal(H4))

        P_inv = torch.inverse(P)
        Lambda_inv = torch.inverse(Lambda)

        A = F.linear(P_inv, Y.T)
        D11 = -F.linear(Lambda_inv, torch.tril(H4, -1).T)
        C1 = F.linear(Lambda_inv, self.Chi)
        Z = -H2 - self.Chi
        B1 = F.linear(P_inv, Z.T)

        return {'P': P, 'A': A, 'B1': B1, 'C1': C1, 'D11': D11}

    def _compute_w(self, xi, u, matrices):
        C1 = matrices['C1']
        D11 = matrices['D11']

        w_list = []
        for i in range(self.nq):
            v = F.linear(xi, C1[i, :]) + self.bv[i] + F.linear(u, self.D12[i, :])
            if i > 0:
                w_so_far = torch.stack(w_list, dim=1)
                v = v + F.linear(w_so_far, D11[i, :i])
            w_i = self.act(v.unsqueeze(1)).squeeze(1)
            w_list.append(w_i)

        return torch.stack(w_list, dim=1)

    def forward(self, t, xi, u):
        matrices = self._compute_constrained_matrices()
        A, B1 = matrices['A'], matrices['B1']
        n_batch = xi.shape[0]

        w = self._compute_w(xi, u, matrices)
        xi_dot = F.linear(xi, A) + F.linear(w, B1) + self.bx.T.expand(n_batch, -1) + F.linear(u, self.B2)

        return xi_dot

    def output(self, xi, u):
        matrices = self._compute_constrained_matrices()
        n_batch = xi.shape[0]

        w = self._compute_w(xi, u, matrices)
        yt = F.linear(xi, self.C2) + F.linear(w, self.D21) + F.linear(u, self.D22) + self.by.T.expand(n_batch, -1)

        return yt


class NODE_REN(nn.Module):
    """Neural ODE with Recurrent Equilibrium Networks for contractive dynamics."""

    def __init__(self, nx, ny, nu, nq, sigma="tanh", epsilon=1.0e-2, device="cpu",
                 bias=False, alpha=0.0, linear_output=False):
        super().__init__()
        self._nfe = 0
        self.sys = _System_contractive(
            nx, ny, nu, nq, sigma, epsilon,
            device=device, bias=bias, linear_output=linear_output, alpha=alpha
        )

    def forward(self, t, x, u):
        self._nfe += 1
        return self.sys(t, x, u)

    def output(self, x, u):
        return self.sys.output(x, u)

    @property
    def nfe(self):
        return self._nfe

    @nfe.setter
    def nfe(self, value):
        self._nfe = value