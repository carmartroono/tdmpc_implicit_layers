"""
world_model.py - TD-MPC2 World Model with DEQ Encoder + NODE-REN Dynamics

Combines:
- DEQ encoder for implicit depth representation learning
- NODE-REN dynamics for contractive state transitions
- 4-phase warmup for stable training
"""

from copy import deepcopy
from enum import IntEnum

import torch
import torch.nn as nn

from common import layers, math, init
from tensordict import TensorDict
from tensordict.nn import TensorDictParams
from implicit_layers.noderen import NODE_REN


class TrainingPhase(IntEnum):
    """Training phases for progressive unfreezing."""
    DYNAMICS_ONLY = 1
    REWARD_CONSISTENCY = 2
    SOFT_JOINT = 3
    FULL_TRAINING = 4


class WorldModel(nn.Module):
    """TD-MPC2 World Model with DEQ encoder and NODE-REN dynamics."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        if cfg.multitask:
            self._task_emb = nn.Embedding(len(cfg.tasks), cfg.task_dim, max_norm=1)
            self.register_buffer("_action_masks", torch.zeros(len(cfg.tasks), cfg.action_dim))
            for i in range(len(cfg.tasks)):
                self._action_masks[i, :cfg.action_dims[i]] = 1.

        # DEQ encoder (implicit depth)
        self._encoder = layers.enc_deq(cfg)

        # NODE-REN dynamics (contractive)
        self._dynamics = NODE_REN(
            nx=cfg.latent_dim,
            ny=cfg.latent_dim,
            nu=cfg.action_dim + cfg.task_dim,
            nq=cfg.noderen_nq,
            sigma=cfg.noderen_sigma,
            epsilon=cfg.noderen_epsilon,
            device=torch.device('cuda:0'),
            bias=cfg.noderen_bias,
            alpha=cfg.noderen_alpha,
            linear_output=cfg.noderen_linear_output
        )

        self._reward = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2 * [cfg.mlp_dim], max(cfg.num_bins, 1))
        self._termination = layers.mlp(cfg.latent_dim + cfg.task_dim, 2 * [cfg.mlp_dim], 1) if cfg.episodic else None
        self._pi = layers.mlp(cfg.latent_dim + cfg.task_dim, 2 * [cfg.mlp_dim], 2 * cfg.action_dim)
        self._Qs = layers.Ensemble([
            layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2 * [cfg.mlp_dim], max(cfg.num_bins, 1), dropout=cfg.dropout)
            for _ in range(cfg.num_q)
        ])

        self.apply(init.weight_init)
        init.zero_([self._reward[-1].weight, self._Qs.params["2", "weight"]])

        self.register_buffer("log_std_min", torch.tensor(cfg.log_std_min))
        self.register_buffer("log_std_dif", torch.tensor(cfg.log_std_max) - self.log_std_min)

        # 4-Phase warmup
        self.phase1_steps = getattr(cfg, 'noderen_phase1_steps', 25000)
        self.phase2_steps = getattr(cfg, 'noderen_phase2_steps', 50000)
        self.phase3_steps = getattr(cfg, 'noderen_phase3_steps', 75000)
        self.current_step = 0
        self._current_phase = TrainingPhase.DYNAMICS_ONLY

        self.dt = cfg.noderen_dt

        self.init()

    def init(self):
        self._detach_Qs_params = TensorDictParams(self._Qs.params.data, no_convert=True)
        self._target_Qs_params = TensorDictParams(self._Qs.params.data.clone(), no_convert=True)

        with self._detach_Qs_params.data.to("meta").to_module(self._Qs.module):
            self._detach_Qs = deepcopy(self._Qs)
            self._target_Qs = deepcopy(self._Qs)

        delattr(self._detach_Qs, "params")
        self._detach_Qs.__dict__["params"] = self._detach_Qs_params
        delattr(self._target_Qs, "params")
        self._target_Qs.__dict__["params"] = self._target_Qs_params

    def __repr__(self):
        s = 'TD-MPC2 World Model (DEQ Encoder + NODE-REN Dynamics)\n'
        modules = ['Encoder', 'Dynamics', 'Reward', 'Termination', 'Policy prior', 'Q-functions']
        for i, m in enumerate([self._encoder, self._dynamics, self._reward, self._termination, self._pi, self._Qs]):
            if m == self._termination and not self.cfg.episodic:
                continue
            s += f"{modules[i]}: {m}\n"
        s += f"Learnable parameters: {self.total_params:,}\n"
        s += f"Phase boundaries: P1={self.phase1_steps}, P2={self.phase2_steps}, P3={self.phase3_steps}"
        return s

    @property
    def total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.init()
        return self

    def train(self, mode=True):
        super().train(mode)
        self._target_Qs.train(False)
        return self

    def soft_update_target_Q(self):
        self._target_Qs_params.lerp_(self._detach_Qs_params, self.cfg.tau)

    def get_training_phase(self):
        if self.current_step < self.phase1_steps:
            return TrainingPhase.DYNAMICS_ONLY
        elif self.current_step < self.phase2_steps:
            return TrainingPhase.REWARD_CONSISTENCY
        elif self.current_step < self.phase3_steps:
            return TrainingPhase.SOFT_JOINT
        return TrainingPhase.FULL_TRAINING

    def get_phase_info(self):
        phase = self.get_training_phase()
        phase_names = {
            TrainingPhase.DYNAMICS_ONLY: "Phase1_DynamicsOnly",
            TrainingPhase.REWARD_CONSISTENCY: "Phase2_RewardConsistency",
            TrainingPhase.SOFT_JOINT: "Phase3_SoftJoint",
            TrainingPhase.FULL_TRAINING: "Phase4_FullTraining"
        }

        if phase == TrainingPhase.DYNAMICS_ONLY:
            progress = self.current_step / self.phase1_steps if self.phase1_steps > 0 else 1.0
        elif phase == TrainingPhase.REWARD_CONSISTENCY:
            progress = (self.current_step - self.phase1_steps) / (self.phase2_steps - self.phase1_steps)
        elif phase == TrainingPhase.SOFT_JOINT:
            progress = (self.current_step - self.phase2_steps) / (self.phase3_steps - self.phase2_steps)
        else:
            progress = 1.0

        return {'phase': int(phase), 'phase_name': phase_names[phase], 'phase_progress': progress, 'current_step': self.current_step}

    def get_loss_weights(self):
        phase = self.get_training_phase()

        if phase == TrainingPhase.DYNAMICS_ONLY:
            return {'consistency': 1.0, 'reward': 0.0, 'value': 0.0, 'policy': 0.0}
        elif phase == TrainingPhase.REWARD_CONSISTENCY:
            progress = (self.current_step - self.phase1_steps) / (self.phase2_steps - self.phase1_steps)
            return {'consistency': 1.0, 'reward': progress, 'value': 0.0, 'policy': 0.0}
        elif phase == TrainingPhase.SOFT_JOINT:
            progress = (self.current_step - self.phase2_steps) / (self.phase3_steps - self.phase2_steps)
            return {'consistency': 1.0, 'reward': 1.0, 'value': progress, 'policy': 0.0}
        return {'consistency': 1.0, 'reward': 1.0, 'value': 1.0, 'policy': 1.0}

    def get_horizon(self):
        phase = self.get_training_phase()
        if phase <= TrainingPhase.SOFT_JOINT:
            return min(2, self.cfg.horizon)
        return self.cfg.horizon

    def should_update_policy(self):
        return self.get_training_phase() == TrainingPhase.FULL_TRAINING

    def increment_step(self):
        self.current_step += 1
        new_phase = self.get_training_phase()
        if new_phase != self._current_phase:
            print(f"[Step {self.current_step}] Transitioning to {new_phase.name}")
            self._current_phase = new_phase

    def task_emb(self, x, task):
        if isinstance(task, int):
            task = torch.tensor([task], device=x.device)
        emb = self._task_emb(task.long())
        if x.ndim == 3:
            emb = emb.unsqueeze(0).repeat(x.shape[0], 1, 1)
        elif emb.shape[0] == 1:
            emb = emb.repeat(x.shape[0], 1)
        return torch.cat([x, emb], dim=-1)

    def encode(self, obs, task):
        if self.cfg.multitask:
            obs = self.task_emb(obs, task)
        if self.cfg.obs == 'rgb' and obs.ndim == 5:
            return torch.stack([self._encoder[self.cfg.obs](o) for o in obs])
        return self._encoder[self.cfg.obs](obs)

    def next(self, z, a, task):
        if self.cfg.multitask:
            z = self.task_emb(z, task)

        if self.cfg.multitask:
            task_emb = self.task_emb(torch.zeros_like(z[:, :1]), task)
            u = torch.cat([a, task_emb], dim=-1)
        else:
            u = torch.cat([a, torch.zeros(z.shape[0], self.cfg.task_dim, device=z.device)], dim=-1)

        xdot = self._dynamics(0.0, z, u)
        z_next = z + self.dt * xdot

        return z_next

    def reward(self, z, a, task):
        if self.cfg.multitask:
            z = self.task_emb(z, task)
        z = torch.cat([z, a], dim=-1)
        return self._reward(z)

    def termination(self, z, task, unnormalized=False):
        assert task is None
        if self.cfg.multitask:
            z = self.task_emb(z, task)
        if unnormalized:
            return self._termination(z)
        return torch.sigmoid(self._termination(z))

    def pi(self, z, task):
        if self.cfg.multitask:
            z = self.task_emb(z, task)

        mean, log_std = self._pi(z).chunk(2, dim=-1)
        log_std = math.log_std(log_std, self.log_std_min, self.log_std_dif)
        eps = torch.randn_like(mean)

        if self.cfg.multitask:
            mean = mean * self._action_masks[task]
            log_std = log_std * self._action_masks[task]
            eps = eps * self._action_masks[task]
            action_dims = self._action_masks.sum(-1)[task].unsqueeze(-1)
        else:
            action_dims = None

        log_prob = math.gaussian_logprob(eps, log_std)
        size = eps.shape[-1] if action_dims is None else action_dims
        scaled_log_prob = log_prob * size

        action = mean + eps * log_std.exp()
        mean, action, log_prob = math.squash(mean, action, log_prob)

        entropy_scale = scaled_log_prob / (log_prob + 1e-8)
        info = TensorDict({
            "mean": mean, "log_std": log_std, "action_prob": 1.,
            "entropy": -log_prob, "scaled_entropy": -log_prob * entropy_scale,
        })
        return action, info

    def Q(self, z, a, task, return_type='min', target=False, detach=False):
        assert return_type in {'min', 'avg', 'all'}

        if self.cfg.multitask:
            z = self.task_emb(z, task)

        z = torch.cat([z, a], dim=-1)
        if target:
            qnet = self._target_Qs
        elif detach:
            qnet = self._detach_Qs
        else:
            qnet = self._Qs
        out = qnet(z)

        if return_type == 'all':
            return out

        qidx = torch.randperm(self.cfg.num_q, device=out.device)[:2]
        Q = math.two_hot_inv(out[qidx], self.cfg)
        if return_type == "min":
            return Q.min(0).values
        return Q.sum(0) / 2