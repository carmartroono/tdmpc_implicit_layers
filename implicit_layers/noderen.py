import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

class _System_contractive(nn.Module):
    def __init__(self, nx, ny, nu, nq, sigma, epsilon, device, bias=False, alpha=0.0, linear_output=False):
        """Used by the upper class NODE_REN to guarantee contractivity to the model. It should not be used by itself.
        Args:
            -nx (int): no. internal-states
            -ny (int): no. output
            -nu (int): no. inputs
            -nq (int): no. non-linear states
            -sigma (string): activation function. It is possible to choose: 'tanh','sigmoid','relu','identity'.
            -epsilon (float): small (positive) scalar used to guarantee the matrices to be positive definitive.
            -device (string): device to be used for the computations using Pytorch (e.g., 'cpu', 'cuda:0',...)
            -bias (bool, optional): choose if the model has non-null biases. Defaults to False.
            -alpha (float, optional): Lower bound of the Contraction rate. If alpha is set to 0, the system continues to be contractive, but with a generic (small) rate. Defaults to 0.
            -linear_output (bool, optional): choose if the output is linear, i.e., choose to force (or not) the matrix D21 to be null. Defaults to False.
        """
        super().__init__()
        # Dimensions of Inputs, Outputs, States
        self.nx = nx  # no. internal-states
        self.ny = ny  # no. output
        self.nu = nu  # no. inputs
        self.nq = nq  # no. non-linear states
        self.epsilon = epsilon
        self.device = device
        std = 1 / np.sqrt(nx)  # Changed initialization std for better conditioning
        # Initialization of the Free Matrices:
        self.Pstar = nn.Parameter(torch.randn(nx, nx, device=device) * std)
        self.Chi = nn.Parameter(torch.randn(nx, nq, device=device) * std)
        # Initialization of the Weights:
        self.Y1 = nn.Parameter(torch.randn(nx, nx, device=device) * std)
        self.B2 = nn.Parameter(torch.randn(nx, nu, device=device) * std)
        self.D12 = nn.Parameter(torch.randn(nq, nu, device=device) * std)
        self.C2 = nn.Parameter(torch.randn(ny, nx, device=device) * std)
        if (linear_output):
            self.D21 = torch.zeros(ny, nq, device=device)
        else:
            self.D21 = nn.Parameter(torch.randn(ny, nq, device=device) * std)
        self.D22 = nn.Parameter(torch.randn(ny, nu, device=device) * std)
        BIAS = bias
        if (BIAS):
            self.bx = nn.Parameter(torch.randn(nx, 1, device=device) * std)
            self.bv = nn.Parameter(torch.randn(nq, 1, device=device) * std)
            self.by = nn.Parameter(torch.randn(ny, 1, device=device) * std)
        else:
            self.bx = torch.zeros(nx, 1, device=device)
            self.bv = torch.zeros(nq, 1, device=device)
            self.by = torch.zeros(ny, 1, device=device)
        self.X = nn.Parameter(
            torch.randn(nx + nq, nx + nq, device=device) * std)  # REMEMBER TO CHANGE IT FOR ROBUST SYSTEMS
        self.alpha = alpha
        # Choosing the activation function:
        if (sigma == "tanh"):
            self.act = nn.Tanh()
        elif (sigma == "sigmoid"):
            self.act = nn.Sigmoid()
        elif (sigma == "relu"):
            self.act = nn.ReLU()
        elif (sigma == "identity"):
            self.act = nn.Identity()
        else:
            print("Error. The chosen sigma function is not valid. Tanh() has been applied.")
            self.act = nn.Tanh()

    def forward(self, t, xi, u):
        # Compute constrained matrices on-the-fly
        P = 0.5 * F.linear(self.Pstar, self.Pstar) + self.epsilon * torch.eye(self.nx, device=self.device)
        H = F.linear(self.X, self.X) + self.epsilon * torch.eye(self.nx + self.nq, device=self.device)
        h1, h2 = torch.split(H, (self.nx, self.nq), dim=0)
        H1, H2 = torch.split(h1, (self.nx, self.nq), dim=1)
        H3, H4 = torch.split(h2, (self.nx, self.nq), dim=1)
        Y = -0.5 * (H1 + self.alpha * P + self.Y1 - self.Y1.T)
        Lambda = 0.5 * torch.diag_embed(torch.diagonal(H4))
        A = F.linear(torch.inverse(P), Y.T)
        D11 = -F.linear(torch.inverse(Lambda), torch.tril(H4, -1).T)
        C1 = F.linear(torch.inverse(Lambda), self.Chi)
        Z = -H2 - self.Chi
        B1 = F.linear(torch.inverse(P), Z.T)

        # Now proceed with the rest of the forward computation
        n_initial_states = xi.shape[0]
        By = F.linear(torch.ones(n_initial_states, 1, device=self.device), self.by)
        vec = torch.zeros(self.nq, 1, device=self.device)
        vec[0, 0] = 1.
        w = torch.zeros(n_initial_states, self.nq, device=self.device)
        v = (F.linear(xi, C1[0, :]) + self.bv[0] * torch.ones(n_initial_states, device=self.device) + F.linear(u, self.D12[0, :])).unsqueeze(1)
        w = w + F.linear(self.act(v), vec)
        for i in range(1, self.nq):
            vec = torch.zeros(self.nq, 1, device=self.device)
            vec[i, 0] = 1.
            v = (F.linear(xi, C1[i, :]) + F.linear(w, D11[i, :]) + self.bv[i] * torch.ones(n_initial_states, device=self.device) + F.linear(u, self.D12[i, :])).unsqueeze(1)
            w = w + F.linear(self.act(v), vec)
        xi_ = F.linear(xi, A) + F.linear(w, B1) + F.linear(torch.ones(n_initial_states, 1, device=self.device), self.bx) + F.linear(u, self.B2)
        return xi_

    def output(self, xi, u):
        # Compute constrained matrices on-the-fly
        P = 0.5 * F.linear(self.Pstar, self.Pstar) + self.epsilon * torch.eye(self.nx, device=self.device)
        H = F.linear(self.X, self.X) + self.epsilon * torch.eye(self.nx + self.nq, device=self.device)
        h1, h2 = torch.split(H, (self.nx, self.nq), dim=0)
        H1, H2 = torch.split(h1, (self.nx, self.nq), dim=1)
        H3, H4 = torch.split(h2, (self.nx, self.nq), dim=1)
        Y = -0.5 * (H1 + self.alpha * P + self.Y1 - self.Y1.T)
        Lambda = 0.5 * torch.diag_embed(torch.diagonal(H4))
        A = F.linear(torch.inverse(P), Y.T)
        D11 = -F.linear(torch.inverse(Lambda), torch.tril(H4, -1).T)
        C1 = F.linear(torch.inverse(Lambda), self.Chi)
        Z = -H2 - self.Chi
        B1 = F.linear(torch.inverse(P), Z.T)
        n_initial_states = xi.shape[0]
        By = F.linear(torch.ones(n_initial_states, 1, device=self.device), self.by)
        vec = torch.zeros(self.nq, 1, device=self.device)
        vec[0, 0] = 1.
        w = torch.zeros(n_initial_states, self.nq, device=self.device)
        v = (F.linear(xi, C1[0, :]) + self.bv[0] * torch.ones(n_initial_states, device=self.device) + F.linear(u, self.D12[0, :])).unsqueeze(1)
        w = w + F.linear(self.act(v), vec)
        for i in range(1, self.nq):
            vec = torch.zeros(self.nq, 1, device=self.device)
            vec[i, 0] = 1.
            v = (F.linear(xi, C1[i, :]) + F.linear(w, D11[i, :]) + self.bv[i] * torch.ones(n_initial_states, device=self.device) + F.linear(u, self.D12[i, :])).unsqueeze(1)
            w = w + F.linear(self.act(v), vec)
        yt = F.linear(xi, self.C2) + F.linear(w, self.D21) + F.linear(u, self.D22) + By
        return yt

    def calculate_w(self, t, xi, u):
        """Calculates the nonlinear feedback w at time t given the state xi and the input u.
        It is used by the module NODE_REN.calculate_w_Vdot_s_vectors().
        """
        # Compute constrained matrices on-the-fly
        P = 0.5 * F.linear(self.Pstar, self.Pstar) + self.epsilon * torch.eye(self.nx, device=self.device)
        H = F.linear(self.X, self.X) + self.epsilon * torch.eye(self.nx + self.nq, device=self.device)
        h1, h2 = torch.split(H, (self.nx, self.nq), dim=0)
        H1, H2 = torch.split(h1, (self.nx, self.nq), dim=1)
        H3, H4 = torch.split(h2, (self.nx, self.nq), dim=1)
        Y = -0.5 * (H1 + self.alpha * P + self.Y1 - self.Y1.T)
        Lambda = 0.5 * torch.diag_embed(torch.diagonal(H4))
        A = F.linear(torch.inverse(P), Y.T)
        D11 = -F.linear(torch.inverse(Lambda), torch.tril(H4, -1).T)
        C1 = F.linear(torch.inverse(Lambda), self.Chi)
        Z = -H2 - self.Chi
        B1 = F.linear(torch.inverse(P), Z.T)

        n_initial_states = xi.shape[0]
        vec = torch.zeros(self.nq, 1, device=self.device)
        vec[0, 0] = 1.
        w = torch.zeros(n_initial_states, self.nq, device=self.device)
        v = (F.linear(xi, C1[0, :]) + self.bv[0] * torch.ones(n_initial_states, device=self.device) + F.linear(u, self.D12[0, :])).unsqueeze(1)
        w = w + F.linear(self.act(v), vec)
        for i in range(1, self.nq):
            vec = torch.zeros(self.nq, 1, device=self.device)
            vec[i, 0] = 1.
            v = (F.linear(xi, C1[i, :]) + F.linear(w, D11[i, :]) + self.bv[i] * torch.ones(n_initial_states, device=self.device) + F.linear(u, self.D12[i, :])).unsqueeze(1)
            w = w + F.linear(self.act(v), vec)
        return w

    def calculate_Vdot(self, delta_x, delta_w, delta_u):
        """Calculates the time-derivative of the storage function V at a given time instant t.
        It is used by the module NODE_REN.calculate_w_Vdot_s_vectors().
        """
        # Compute constrained matrices on-the-fly
        P = 0.5 * F.linear(self.Pstar, self.Pstar) + self.epsilon * torch.eye(self.nx, device=self.device)
        H = F.linear(self.X, self.X) + self.epsilon * torch.eye(self.nx + self.nq, device=self.device)
        h1, h2 = torch.split(H, (self.nx, self.nq), dim=0)
        H1, H2 = torch.split(h1, (self.nx, self.nq), dim=1)
        H3, H4 = torch.split(h2, (self.nx, self.nq), dim=1)
        Y = -0.5 * (H1 + self.alpha * P + self.Y1 - self.Y1.T)
        Lambda = 0.5 * torch.diag_embed(torch.diagonal(H4))
        A = F.linear(torch.inverse(P), Y.T)
        D11 = -F.linear(torch.inverse(Lambda), torch.tril(H4, -1).T)
        C1 = F.linear(torch.inverse(Lambda), self.Chi)
        Z = -H2 - self.Chi
        B1 = F.linear(torch.inverse(P), Z.T)

        delta_xdot = F.linear(delta_x, A) + F.linear(delta_w, B1) + F.linear(torch.ones(1, 1), self.bx) + F.linear(delta_u, self.B2)
        Vdot = F.linear(F.linear(delta_xdot, P.T), delta_x) + F.linear(F.linear(delta_x, P.T), delta_xdot)
        return Vdot

    def calculate_s(self, delta_y, delta_u):
        """(Dummy function)
        """
        return 0

class NODE_REN(nn.Module):
    def __init__(self, nx, ny, nu, nq, sigma="tanh", epsilon=1.0e-2, device="cpu", bias=False, alpha=0.0,
                 linear_output=False):
        super().__init__()
        self.nfe = 0
        self.sys = _System_contractive(nx, ny, nu, nq, sigma, epsilon, device=device, bias=bias,
                                       linear_output=linear_output, alpha=alpha)

    def forward(self, t, x, u):
        self.nfe += 1
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        if torch.isnan(u).any() or torch.isinf(u).any():
            u = torch.nan_to_num(u, nan=0.0, posinf=1.0, neginf=-1.0)

        xdot = self.sys(t, x, u)

        #xdot = torch.clamp(xdot, min=-10.0, max=10.0)
        xdot = torch.nan_to_num(xdot, nan=0.0, posinf=1e6, neginf=-1e6)
        return xdot

    def output(self, x, u):
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        if torch.isnan(u).any() or torch.isinf(u).any():
            u = torch.nan_to_num(u, nan=0.0, posinf=1.0, neginf=-1.0)

        yt = self.sys.output(x, u)
        yt = torch.nan_to_num(yt, nan=0.0, posinf=1e6, neginf=-1e6)
        return yt

    @property
    def nfe(self):
        return self._nfe

    @nfe.setter
    def nfe(self, value):
        self._nfe = value