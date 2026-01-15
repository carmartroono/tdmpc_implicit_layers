"""
tdmpc2.py - TD-MPC2 agent with DEQ Encoder + NODE-REN Dynamics

Features:
- DEQ encoder for implicit depth representation
- NODE-REN dynamics for contractive transitions
- 4-phase warmup for stable training
- Phase-aware loss weighting and horizon
"""

import torch
import torch.nn.functional as F

from common import math
from common.scale import RunningScale
from common.world_model import WorldModel
from common.layers import api_model_conversion
from tensordict import TensorDict
import os

os.environ["HYDRA_FULL_ERROR"] = "1"


class TDMPC2(torch.nn.Module):
    """TD-MPC2 agent with DEQ encoder and NODE-REN dynamics."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device('cuda:0')
        self.model = WorldModel(cfg).to(self.device)

        self.optim = torch.optim.Adam([
            {'params': self.model._encoder.parameters(), 'lr': self.cfg.lr * self.cfg.enc_lr_scale},
            {'params': self.model._reward.parameters()},
            {'params': self.model._termination.parameters() if self.cfg.episodic else []},
            {'params': self.model._Qs.parameters()},
            {'params': self.model._task_emb.parameters() if self.cfg.multitask else []},
            {'params': self.model._dynamics.parameters(), 'lr': self.cfg.lr * self.cfg.noderen_lr_scale}
        ], lr=self.cfg.lr, capturable=True)

        self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr, eps=1e-5, capturable=True)

        self.model.eval()
        self.scale = RunningScale(cfg)
        self.cfg.iterations += 2 * int(cfg.action_dim >= 20)

        self.discount = torch.tensor(
            [self._get_discount(ep_len) for ep_len in cfg.episode_lengths], device='cuda:0'
        ) if self.cfg.multitask else self._get_discount(cfg.episode_length)

        print('Episode length:', cfg.episode_length)
        print('Discount factor:', self.discount)

        self._prev_mean = torch.nn.Buffer(torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device))

        if cfg.compile:
            print('Compiling update function with torch.compile...')
            self._update = torch.compile(self._update, mode="reduce-overhead")

    @property
    def plan(self):
        _plan_val = getattr(self, "_plan_val", None)
        if _plan_val is not None:
            return _plan_val
        if self.cfg.compile:
            plan = torch.compile(self._plan, mode="reduce-overhead")
        else:
            plan = self._plan
        self._plan_val = plan
        return self._plan_val

    def _get_discount(self, episode_length):
        frac = episode_length / self.cfg.discount_denom
        return min(max((frac - 1) / frac, self.cfg.discount_min), self.cfg.discount_max)

    def save(self, fp):
        torch.save({"model": self.model.state_dict()}, fp)

    def load(self, fp):
        if isinstance(fp, dict):
            state_dict = fp
        else:
            state_dict = torch.load(fp, map_location=torch.get_default_device(), weights_only=False)
        state_dict = state_dict["model"] if "model" in state_dict else state_dict
        state_dict = api_model_conversion(self.model.state_dict(), state_dict)
        self.model.load_state_dict(state_dict)

    @torch.no_grad()
    def act(self, obs, t0=False, eval_mode=False, task=None):
        obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
        if task is not None:
            task = torch.tensor([task], device=self.device)
        if self.cfg.mpc:
            return self.plan(obs, t0=t0, eval_mode=eval_mode, task=task).cpu()
        z = self.model.encode(obs, task)
        action, info = self.model.pi(z, task)
        if eval_mode:
            action = info["mean"]
        return action[0].cpu()

    @torch.no_grad()
    def _estimate_value(self, z, actions, task):
        G, discount = 0, 1
        horizon = self.model.get_horizon()
        termination = torch.zeros(self.cfg.num_samples, 1, dtype=torch.float32, device=z.device)

        for t in range(min(horizon, actions.shape[0])):
            reward = math.two_hot_inv(self.model.reward(z, actions[t], task), self.cfg)
            z = self.model.next(z, actions[t], task)
            G = G + discount * (1 - termination) * reward
            discount_update = self.discount[torch.tensor(task)] if self.cfg.multitask else self.discount
            discount = discount * discount_update
            if self.cfg.episodic:
                termination = torch.clip(termination + (self.model.termination(z, task) > 0.5).float(), max=1.)

        action, _ = self.model.pi(z, task)
        return G + discount * (1 - termination) * self.model.Q(z, action, task, return_type='avg')

    @torch.no_grad()
    def _plan(self, obs, t0=False, eval_mode=False, task=None):
        horizon = self.model.get_horizon()

        z = self.model.encode(obs, task)
        if self.cfg.num_pi_trajs > 0:
            pi_actions = torch.empty(horizon, self.cfg.num_pi_trajs, self.cfg.action_dim, device=self.device)
            _z = z.repeat(self.cfg.num_pi_trajs, 1)
            for t in range(horizon - 1):
                pi_actions[t], _ = self.model.pi(_z, task)
                _z = self.model.next(_z, pi_actions[t], task)
            pi_actions[-1], _ = self.model.pi(_z, task)

        z = z.repeat(self.cfg.num_samples, 1)
        mean = torch.zeros(horizon, self.cfg.action_dim, device=self.device)
        std = torch.full((horizon, self.cfg.action_dim), self.cfg.max_std, dtype=torch.float, device=self.device)
        if not t0:
            mean[:min(horizon, self.cfg.horizon) - 1] = self._prev_mean[1:min(horizon, self.cfg.horizon)]
        actions = torch.empty(horizon, self.cfg.num_samples, self.cfg.action_dim, device=self.device)
        if self.cfg.num_pi_trajs > 0:
            actions[:, :self.cfg.num_pi_trajs] = pi_actions

        for _ in range(self.cfg.iterations):
            r = torch.randn(horizon, self.cfg.num_samples - self.cfg.num_pi_trajs, self.cfg.action_dim, device=std.device)
            actions_sample = mean.unsqueeze(1) + std.unsqueeze(1) * r
            actions_sample = actions_sample.clamp(-1, 1)
            actions[:, self.cfg.num_pi_trajs:] = actions_sample
            if self.cfg.multitask:
                actions = actions * self.model._action_masks[task]

            value = self._estimate_value(z, actions, task).nan_to_num(0)
            elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices
            elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

            max_value = elite_value.max(0).values
            score = torch.exp(self.cfg.temperature * (elite_value - max_value))
            score = score / score.sum(0)
            mean = (score.unsqueeze(0) * elite_actions).sum(dim=1) / (score.sum(0) + 1e-9)
            std = ((score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2).sum(dim=1) / (score.sum(0) + 1e-9)).sqrt()
            std = std.clamp(self.cfg.min_std, self.cfg.max_std)
            if self.cfg.multitask:
                mean = mean * self.model._action_masks[task]
                std = std * self.model._action_masks[task]

        rand_idx = math.gumbel_softmax_sample(score.squeeze(1))
        actions = torch.index_select(elite_actions, 1, rand_idx).squeeze(1)
        a, std = actions[0], std[0]
        if not eval_mode:
            a = a + std * torch.randn(self.cfg.action_dim, device=std.device)

        self._prev_mean[:horizon].copy_(mean[:horizon])
        if horizon < self.cfg.horizon:
            self._prev_mean[horizon:].zero_()

        return a.clamp(-1, 1)

    def update_pi(self, zs, task):
        if not self.model.should_update_policy():
            return TensorDict({
                "pi_loss": torch.tensor(0.0),
                "pi_grad_norm": torch.tensor(0.0),
                "pi_entropy": torch.tensor(0.0),
                "pi_scaled_entropy": torch.tensor(0.0),
                "pi_scale": self.scale.value,
            })

        action, info = self.model.pi(zs, task)
        qs = self.model.Q(zs, action, task, return_type='avg', detach=True)
        self.scale.update(qs[0])
        qs = self.scale(qs)

        rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
        pi_loss = (-(self.cfg.entropy_coef * info["scaled_entropy"] + qs).mean(dim=(1, 2)) * rho).mean()
        pi_loss.backward()
        pi_grad_norm = torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm)
        self.pi_optim.step()
        self.pi_optim.zero_grad(set_to_none=True)

        return TensorDict({
            "pi_loss": pi_loss,
            "pi_grad_norm": pi_grad_norm,
            "pi_entropy": info["entropy"],
            "pi_scaled_entropy": info["scaled_entropy"],
            "pi_scale": self.scale.value,
        })

    @torch.no_grad()
    def _td_target(self, next_z, reward, terminated, task):
        action, _ = self.model.pi(next_z, task)
        discount = self.discount[task].unsqueeze(-1) if self.cfg.multitask else self.discount
        return reward + discount * (1 - terminated) * self.model.Q(next_z, action, task, return_type='min', target=True)

    def _update(self, obs, action, reward, terminated, task=None):
        loss_weights = self.model.get_loss_weights()
        horizon = self.model.get_horizon()

        with torch.no_grad():
            next_z = self.model.encode(obs[1:], task)
            td_targets = self._td_target(next_z, reward, terminated, task)

        self.model.train()

        effective_horizon = min(horizon, self.cfg.horizon)
        zs = torch.empty(effective_horizon + 1, self.cfg.batch_size, self.cfg.latent_dim, device=self.device)
        z = self.model.encode(obs[0], task)
        zs[0] = z

        consistency_loss = 0
        for t in range(effective_horizon):
            _action = action[t] if t < action.shape[0] else action[-1]
            _next_z = next_z[t] if t < next_z.shape[0] else next_z[-1]
            z = self.model.next(z, _action, task)
            consistency_loss = consistency_loss + F.mse_loss(z, _next_z) * self.cfg.rho ** t
            zs[t + 1] = z

        _zs = zs[:-1]
        qs = self.model.Q(_zs, action[:effective_horizon], task, return_type='all')
        reward_preds = self.model.reward(_zs, action[:effective_horizon], task)
        if self.cfg.episodic:
            termination_pred = self.model.termination(zs[1:], task, unnormalized=True)

        reward_loss, value_loss = 0, 0
        for t in range(effective_horizon):
            rew_pred = reward_preds[t]
            rew_target = reward[t] if t < reward.shape[0] else reward[-1]
            td_target = td_targets[t] if t < td_targets.shape[0] else td_targets[-1]
            qs_t = qs[:, t]

            reward_loss = reward_loss + math.soft_ce(rew_pred, rew_target, self.cfg).mean() * self.cfg.rho ** t
            for q in qs_t:
                value_loss = value_loss + math.soft_ce(q, td_target, self.cfg).mean() * self.cfg.rho ** t

        consistency_loss = consistency_loss / effective_horizon
        reward_loss = reward_loss / effective_horizon
        if self.cfg.episodic:
            termination_loss = F.binary_cross_entropy_with_logits(termination_pred, terminated[:effective_horizon])
        else:
            termination_loss = 0.
        value_loss = value_loss / (effective_horizon * self.cfg.num_q)

        total_loss = (
            self.cfg.consistency_coef * loss_weights['consistency'] * consistency_loss +
            self.cfg.reward_coef * loss_weights['reward'] * reward_loss +
            self.cfg.termination_coef * termination_loss +
            self.cfg.value_coef * loss_weights['value'] * value_loss
        )

        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
        self.optim.step()
        self.optim.zero_grad(set_to_none=True)

        pi_info = self.update_pi(zs.detach(), task)
        self.model.soft_update_target_Q()
        self.model.increment_step()

        self.model.eval()

        phase_info = self.model.get_phase_info()
        info = TensorDict({
            "consistency_loss": consistency_loss,
            "reward_loss": reward_loss,
            "value_loss": value_loss,
            "termination_loss": termination_loss,
            "total_loss": total_loss,
            "grad_norm": grad_norm,
            "training_phase": float(phase_info['phase']),
            "phase_progress": float(phase_info['phase_progress']),
            "current_step": float(phase_info['current_step']),
            "effective_horizon": float(effective_horizon),
            "dynamics_nfe": float(self.model._dynamics.nfe if hasattr(self.model._dynamics, 'nfe') else 0),
        })

        if self.cfg.episodic:
            info.update(math.termination_statistics(torch.sigmoid(termination_pred[-1]), terminated[-1]))
        info.update(pi_info)

        return info.detach().mean()

    def update(self, buffer):
        obs, action, reward, terminated, task = buffer.sample()
        kwargs = {}
        if task is not None:
            kwargs["task"] = task
        torch.compiler.cudagraph_mark_step_begin()
        return self._update(obs, action, reward, terminated, **kwargs)