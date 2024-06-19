import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import copy
import wandb
from collections import defaultdict
import torch.nn.functional as F
from diffusion_DICE import dpm_solver_pytorch
from diffusion_DICE import schedule
from scipy.special import softmax
import math

EXP_ADV_MAX = 100.
EXP_ADV_MIN = 1e-40
ADV_MIN = -20.
EXP_SP_MAX = 5.

def update_target(new, target, tau):
    # Update the frozen target models
    for param, target_param in zip(new.parameters(), target.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

class GaussianFourierProjection(nn.Module):
  """Gaussian random features for encoding time steps."""  
  def __init__(self, embed_dim, scale=30.):
    super().__init__()
    # Randomly sample weights during initialization. These weights are fixed 
    # during optimization and are not trainable.
    self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
  def forward(self, x):
    x_proj = x[..., None] * self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class Dense(nn.Module):
  """A fully connected layer that reshapes outputs to feature maps."""
  def __init__(self, input_dim, output_dim):
    super().__init__()
    self.dense = nn.Linear(input_dim, output_dim)
  def forward(self, x):
    return self.dense(x)

class SiLU(nn.Module):
  def __init__(self):
    super().__init__()
  def forward(self, x):
    return x * torch.sigmoid(x)

class Squeeze(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(dim=self.dim)

def mlp(dims, activation=nn.ReLU, output_activation=None, layer_norm=False, dropout_rate=0.0, squeeze_output=False):
    n_dims = len(dims)
    assert n_dims >= 2, 'MLP requires at least two dims (input and output)'
    layers = []
    for i in range(n_dims - 2):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        if layer_norm:
            layers.append(nn.LayerNorm(dims[i+1]))
        if dropout_rate > 0:
            layers.append(nn.Dropout(p=dropout_rate))
        layers.append(activation())
    layers.append(nn.Linear(dims[-2], dims[-1]))
    if output_activation is not None:
        layers.append(output_activation())
    if squeeze_output:
        # assert dims[-1] == 1
        layers.append(Squeeze(-1))
    net = nn.Sequential(*layers)
    net.to(dtype=torch.float32)
    return net

class VectorizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, ensemble_size: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size

        self.weight = nn.Parameter(torch.empty(ensemble_size, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(ensemble_size, 1, out_features))

        self.reset_parameters()

    def reset_parameters(self):
        # default pytorch init for nn.Linear module
        for layer in range(self.ensemble_size):
            nn.init.kaiming_uniform_(self.weight[layer], a=math.sqrt(5))

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input: [ensemble_size, batch_size, input_size]
        # weight: [ensemble_size, input_size, out_size]
        # out: [ensemble_size, batch_size, out_size]
        return x @ self.weight + self.bias
    
class VectorizedCritic(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int, num_critics: int
    ):
        super().__init__()
        self.critic = nn.Sequential(
            VectorizedLinear(state_dim + action_dim, hidden_dim, num_critics),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, hidden_dim, num_critics),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, hidden_dim, num_critics),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, 1, num_critics),
        )
        # init as in the EDAC paper
        for layer in self.critic[::2]:
            torch.nn.init.constant_(layer.bias, 0.1)

        torch.nn.init.uniform_(self.critic[-1].weight, -3e-3, 3e-3)
        torch.nn.init.uniform_(self.critic[-1].bias, -3e-3, 3e-3)

        self.num_critics = num_critics

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # [..., batch_size, state_dim + action_dim]
        state_action = torch.cat([state, action], dim=-1)
        if state_action.dim() != 3:
            assert state_action.dim() == 2
            # [num_critics, batch_size, state_dim + action_dim]
            state_action = state_action.unsqueeze(0).repeat_interleave(
                self.num_critics, dim=0
            )
        assert state_action.dim() == 3
        assert state_action.shape[0] == self.num_critics
        # [num_critics, batch_size]
        q_values = self.critic(state_action).squeeze(-1)
        return q_values
    
class VectorizedQ(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int, num_critics: int
    ):
        super().__init__()
        self.critic = nn.Sequential(
            VectorizedLinear(state_dim + action_dim, hidden_dim, num_critics),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, hidden_dim, num_critics),
            # nn.ReLU(),
            # VectorizedLinear(hidden_dim, hidden_dim, num_critics),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, 1, num_critics),
        )

        self.num_critics = num_critics

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # [..., batch_size, state_dim + action_dim]
        state_action = torch.cat([state, action], dim=-1)
        if state_action.dim() != 3:
            assert state_action.dim() == 2
            # [num_critics, batch_size, state_dim + action_dim]
            state_action = state_action.unsqueeze(0).repeat_interleave(
                self.num_critics, dim=0
            )
        assert state_action.dim() == 3
        assert state_action.shape[0] == self.num_critics
        # [num_critics, batch_size]
        q_values = self.critic(state_action).squeeze(-1)
        return q_values

class Residual_Block(nn.Module):
    def __init__(self, input_dim, output_dim, t_dim=128, last=False):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SiLU(),
            nn.Linear(t_dim, output_dim),
        )
        self.dense1 = nn.Sequential(nn.Linear(input_dim, output_dim),SiLU())
        self.dense2 = nn.Sequential(nn.Linear(output_dim, output_dim),SiLU())
        self.modify_x = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
    def forward(self, x, t):
        h1 = self.dense1(x) + self.time_mlp(t)
        h2 = self.dense2(h1)
        return h2 + self.modify_x(x)

class TwinQ(nn.Module):
    def __init__(self, action_dim, state_dim, layer_norm=False, squeeze_output=False, hidden_dim=256, n_hidden=2):
        super().__init__()
        dims = [state_dim + action_dim, *([hidden_dim] * n_hidden), 1]
        self.q1 = mlp(dims, layer_norm=layer_norm, squeeze_output=squeeze_output)
        self.q2 = mlp(dims, layer_norm=layer_norm, squeeze_output=squeeze_output)

    def both(self, action, condition=None):
        as_ = torch.cat([action, condition], -1) if condition is not None else action
        return self.q1(as_), self.q2(as_)
    
    def first(self, action, condition=None):
        as_ = torch.cat([action, condition], -1) if condition is not None else action
        return self.q1(as_)
    
    def second(self, action, condition=None):
        as_ = torch.cat([action, condition], -1) if condition is not None else action
        return self.q2(as_)

    def forward(self, action, condition=None):
        return torch.min(*self.both(action, condition))
    
class ValueFunction(nn.Module):
    def __init__(self, state_dim, layer_norm=False, dropout_rate=0.0, hidden_dim=256, n_hidden=2):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v = mlp(dims, layer_norm=layer_norm, dropout_rate=dropout_rate, squeeze_output=True)

    def forward(self, state):
        return self.v(state)

class TwinV(nn.Module):
    def __init__(self, state_dim, layer_norm=False, dropout_rate=0.0, hidden_dim=256, n_hidden=2):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v1 = mlp(dims, layer_norm=layer_norm, dropout_rate=dropout_rate, squeeze_output=True)
        self.v2 = mlp(dims, layer_norm=layer_norm, dropout_rate=dropout_rate, squeeze_output=True)

    def both(self, state):
        return torch.stack([self.v1(state), self.v2(state)], dim=0)

    def forward(self, state):
        return torch.min(self.both(state), dim=0)[0]

class GuidanceWt(nn.Module):
    def __init__(self, action_dim, state_dim, hidden_dim=256, n_hidden=4):
        super().__init__()
        dims = [action_dim+32+state_dim, *([hidden_dim] * n_hidden), 1]
        self.wt = mlp(dims, activation=SiLU, squeeze_output=True)
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=32), nn.Linear(32, 32))
        
    def forward(self, action, t, condition=None):
        embed = self.embed(t)
        ats = torch.cat([action, embed, condition], -1) if condition is not None else torch.cat([action, embed], -1)
        return self.wt(ats)

class Critic_Guide(nn.Module):
    def __init__(self, adim, sdim) -> None:
        super().__init__()
        self.conditional_sampling = False if sdim==0 else True
        self.v0 = None
        self.wt = None

    def forward(self, condition):
        return self.v0(condition)

    def calculate_guidance(self, a, t, condition=None):
        raise NotImplementedError
    
    def calculateV(self, condition):
        return self(condition)
    
    def update_v0(self, data):
        raise NotImplementedError
    
    def update_wt(self, data):
        raise NotImplementedError

class SDICE_Critic(Critic_Guide):
    def __init__(self, adim, sdim, args) -> None:
        super().__init__(adim, sdim)
        self.q0 = TwinQ(adim, sdim, layer_norm=False, squeeze_output=True).to(args.device)
        self.q0_target = copy.deepcopy(self.q0).requires_grad_(False).to(args.device)
        self.v0 = ValueFunction(sdim, layer_norm=True).to(args.device)
        if args.q_ensemble_num > 0:
            self.q_ensemble = VectorizedQ(sdim, adim, 256, args.q_ensemble_num)
            self.q_ensemble_optimizer = torch.optim.AdamW(self.q_ensemble.parameters(), lr=args.value_lr, weight_decay=args.weight_decay)
        self.wt = GuidanceWt(adim, sdim, hidden_dim=args.hidden_dim).to(args.device)
        self.q_optimizer = torch.optim.AdamW(self.q0.parameters(), lr=args.value_lr, weight_decay=args.weight_decay)
        self.v_optimizer = torch.optim.AdamW(self.v0.parameters(), lr=args.value_lr, weight_decay=args.weight_decay)
        self.wt_optimizer = torch.optim.AdamW(self.wt.parameters(), lr=args.wt_lr, weight_decay=args.weight_decay)
        if args.use_lr_schedule:
            from torch.optim.lr_scheduler import CosineAnnealingLR
            self.schedulers = []
            if args.q_ensemble_num > 0:
                self.schedulers.append(CosineAnnealingLR(self.q_ensemble_optimizer, T_max=args.train_epoch, eta_min=args.min_value_lr))
            self.schedulers.append(CosineAnnealingLR(self.q_optimizer, T_max=args.train_epoch, eta_min=args.min_value_lr))
            self.schedulers.append(CosineAnnealingLR(self.v_optimizer, T_max=args.train_epoch, eta_min=args.min_value_lr))
        self.discount = 0.99
        self.args = args
        self.M = args.M
        self.alpha = args.alpha
        self.guidance_scale = 1.0
        self.v0_step = 0
        self.wt_step = 0
        self.info = defaultdict(lambda : [])

    def calculate_guidance(self, a, t, condition=None):
        with torch.enable_grad():
            a.requires_grad_(True)
            W_t = self.wt(a, t, condition)
            guidance =  self.guidance_scale * torch.autograd.grad(torch.sum(W_t), a)[0]
        return guidance.detach()
    
    def perturb_fake_a(self, fake_a):
        random_t = torch.rand((fake_a.shape[0], ), device=fake_a.device) * (1. - 1e-3) + 1e-3
        random_t = torch.stack([random_t] * fake_a.shape[1], dim=1)
        z = torch.randn_like(fake_a)
        alpha_t, std = schedule.marginal_prob_std(random_t)
        perturbed_fake_a = fake_a * alpha_t[..., None] + z * std[..., None]
        return perturbed_fake_a, random_t

    def update_v0(self, data):
        s = data["s"]
        a = data["a"]
        r = data["r"]
        s_ = data["s_"]
        d = data["d"]
        
        # update v0
        with torch.no_grad():
            target_q = self.q0_target(a, s)
        v = self.v0(s)
        
        sp_term = (target_q - v) / self.alpha
        clipped_sp_term = torch.clamp(sp_term, max=1.0) # here 1.0 is just a manually selected constant to avoid overflow during back propagation
        td_mean, td_min, td_max = sp_term.mean(), sp_term.min(), sp_term.max()
        # piecewise f-divergence
        residual_loss = torch.where(sp_term >= 0, sp_term**2 / 4 + sp_term, torch.exp(clipped_sp_term) - 1)
        value_loss = torch.mean(residual_loss + v / self.alpha)

        self.v_optimizer.zero_grad(set_to_none=True)
        value_loss.backward()
        self.v_optimizer.step()

        # update q0
        with torch.no_grad():
            next_v = self.v0(s_)
        target_q = r + (1. - d.float()) * self.discount * next_v
        q1, q2 = self.q0.both(a, s)
        critic_loss = torch.mean((target_q - q1)**2 + (target_q - q2)**2)

        self.q_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.q_optimizer.step()
        
        # update ensemble q if needed
        if self.args.q_ensemble_num > 0:
            independent_q = self.q_ensemble(s, a) # ensemble_num, bz
            q_ensemble_loss = torch.mean((target_q.unsqueeze(0) - independent_q)**2)
            self.q_ensemble_optimizer.zero_grad(set_to_none=True)
            q_ensemble_loss.backward()
            self.q_ensemble_optimizer.step()

        # update target q0
        update_target(self.q0, self.q0_target, 0.005)
        
        if (self.v0_step + 1) % 1000 == 0:
            wandb.log({"td_mean": np.mean(self.info["td_mean"]), "td_min": np.mean(self.info["td_min"]),
                        "td_max": np.mean(self.info["td_max"]), "v_value": np.mean(self.info["v_value"]),
                        "ensemble_loss": np.mean(self.info["ensemble_loss"]), }, step=self.v0_step)
            self.info["v_value"].clear()
            for key in ["mean", "min", "max"]:
                self.info[f"td_{key}"].clear()
        self.v0_step += 1
        self.info["v_value"].append(torch.mean(v).item())
        self.info["td_mean"].append(td_mean.item())
        self.info["td_min"].append(td_min.item())
        self.info["td_max"].append(td_max.item())
        
    def update_wt(self, data):
        # input  many s <bz, S>  action <bz, M, A>,
        s = data['s']
        a = data['a']
        fake_a = a.unsqueeze(1).expand([a.shape[0], self.M, a.shape[1]])
        
        random_t = torch.rand((fake_a.shape[0], ), device=s.device) * (1. - 1e-3) + 1e-3
        random_t = torch.stack([random_t] * fake_a.shape[1], dim=1)
        z = torch.randn_like(fake_a)
        alpha_t, std = schedule.marginal_prob_std(random_t)
        perturbed_fake_a = fake_a * alpha_t[..., None] + z * std[..., None]
        pred_energy = self.wt(perturbed_fake_a, random_t, torch.stack([s]*fake_a.shape[1] ,axis=1))

        dual_loss = self.independent_clip_dual_loss(s, a, pred_energy)
        
        self.wt_optimizer.zero_grad(set_to_none=True)
        dual_loss.backward()
        self.wt_optimizer.step()

        if (self.wt_step + 1) % 1000 == 0:
            wandb.log({"energy_mean": np.mean(self.info["energy_mean"]), "energy_min": np.mean(self.info["energy_min"]),
                        "energy_max": np.mean(self.info["energy_max"])}, step=self.wt_step)
            for key in ["mean", "min", "max"]:
                self.info[f"energy_{key}"].clear()
        self.wt_step += 1
        self.info["energy_mean"].append(torch.mean(pred_energy).item())
        self.info["energy_min"].append(torch.min(pred_energy).item())
        self.info["energy_max"].append(torch.max(pred_energy).item())

        return dual_loss.detach().cpu().numpy()

    def independent_clip_dual_loss(self, s, a, pred_energy):
        with torch.no_grad():
            target_q = self.q0_target(a, s)
            v = self.v0(s)
            pi_residual = (target_q - v) / self.alpha
            if pi_residual.dim() == 1: # bz -> (bz, 1)
                pi_residual = pi_residual.unsqueeze(1)
            weight = torch.where(pi_residual >= 0, pi_residual / 2 + 1, torch.exp(pi_residual))
        weight = torch.clamp(weight, min=EXP_ADV_MIN, max=EXP_ADV_MAX).detach()

        dual_loss = self.clip_normalize_dual_loss(weight, pred_energy)
        return dual_loss
    
    def clip_normalize_dual_loss(self, weight, energy):
        clipped_energy_target = torch.minimum(torch.zeros_like(energy), energy + 80.0).detach()
        sp_term = clipped_energy_target - energy
        max_sp_term = torch.max(sp_term)
        max_sp_term = torch.where(max_sp_term < -1.0, -1.0, max_sp_term).detach()
        dual_loss = torch.mean(weight * torch.exp(sp_term - max_sp_term) + torch.exp(-max_sp_term) * energy)
        return dual_loss


class ScoreBase(nn.Module):
    def __init__(self, input_dim, output_dim, marginal_prob_std, embed_dim=32, args=None):
        super().__init__()
        self.output_dim = output_dim
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim))
        self.device=args.device
        self.noise_schedule = dpm_solver_pytorch.NoiseScheduleVP(schedule='linear')
        self.dpm_solver = dpm_solver_pytorch.DPM_Solver(self.forward_dmp_wrapper_fn, self.noise_schedule, predict_x0=True)
        self.marginal_prob_std = marginal_prob_std
        self.v = []
        self.v.append(SDICE_Critic(adim=output_dim, sdim=input_dim-output_dim, args=args))
        self.args = args
        self.inference_sample = args.inference_sample

    def forward_dmp_wrapper_fn(self, x, t):
        score = self(x, t)
        result = - (score + self.v[0].calculate_guidance(x, t, self.condition)) * self.marginal_prob_std(t)[1][..., None]
        return result
    
    def dpm_wrapper_sample(self, dim, batch_size, **kwargs):
        with torch.no_grad():
            init_x = torch.randn(batch_size, dim, device=self.device)
            return self.dpm_solver.sample(init_x, **kwargs).cpu().numpy()

    def calculateV(self, s):
        assert s != None
        return self.v[0](s)
    
    def forward(self, x, t, condition=None):
        raise NotImplementedError

    def select_actions(self, states, diffusion_steps=15):
        self.eval()
        multiple_input=True
        with torch.no_grad():
            states = torch.FloatTensor(states).to(self.device)
            if states.dim == 1:
                states = states.unsqueeze(0)
                multiple_input=False
            if self.inference_sample > 1:
                states = states.repeat_interleave(self.inference_sample, dim=0)
            num_states = states.shape[0]
            self.condition = states
            results = self.dpm_wrapper_sample(self.output_dim, batch_size=states.shape[0], steps=diffusion_steps, order=2)
            actions = results.reshape(num_states, self.output_dim).copy() # <bz, A>
            if self.inference_sample > 1:
                actions_tensor = torch.tensor(actions).to(self.device)
                actions = actions.reshape(int(num_states / self.inference_sample), self.inference_sample, self.output_dim) # bz, sample, act_dim
                if self.args.q_ensemble_num > 0:
                    max_certainty_idx = self.filter_by_independent_critic(states, actions_tensor) # actoins_tensor shape is bz * sample, act_dim
                else:
                    max_certainty_idx = self.filter_by_critic(states, actions_tensor) # actoins_tensor shape is bz * sample, act_dim
            self.condition = None
        if self.inference_sample > 1:
            out_actions = [actions[i, max_certainty_idx[i]] for i in range(actions.shape[0])] if multiple_input else actions[0, max_certainty_idx[0]]
        else:
            out_actions = [actions[i] for i in range(actions.shape[0])] if multiple_input else actions[0]
        self.train()
        return out_actions
    
    def filter_by_critic(self, states, actions):
        with torch.no_grad():
            target_q = self.v[0].q0_target.first(actions, states) # bz * sample
        target_q = target_q.reshape(int(target_q.shape[0] / self.inference_sample), self.inference_sample) # bz, sample
        if self.args.filter_type == "max":
            max_q_val, max_q_idx = torch.max(target_q, dim=-1)
        elif self.args.filter_type == "softmax":
            softmax_logits = torch.exp((target_q- target_q.max(dim=1, keepdim=True)[0]))
            softmax_dists = [Categorical(probs=row) for row in softmax_logits]
            max_q_idx = torch.stack([dist.sample() for dist in softmax_dists])
            max_q_val = torch.stack([target_q[i, max_q_idx[i]] for i in range(target_q.shape[0])])
        else:
            raise ValueError("Invalid filter type!!!")

        self.v[0].info["q_of_chosen_action"].append(torch.mean(max_q_val).item())
        return max_q_idx
    
    def filter_by_independent_critic(self, states, actions):
        with torch.no_grad():
            ensembled_q = self.v[0].q_ensemble(states, actions) # ensemble_num, bz * sample
        q_mean = torch.mean(ensembled_q.reshape(ensembled_q.shape[0], int(ensembled_q.shape[1] / self.inference_sample), self.inference_sample), dim=0) # ensemble_num, bz, sample
        if self.args.filter_type == "max":
            max_q_mean_val, max_q_mean_idx = torch.max(q_mean, dim=-1)
        elif self.args.filter_type == "softmax":
            softmax_logits = torch.exp((q_mean- q_mean.max(dim=1, keepdim=True)[0]))
            softmax_dists = [Categorical(probs=row) for row in softmax_logits]
            max_q_mean_idx = torch.stack([dist.sample() for dist in softmax_dists])
            max_q_mean_val = torch.stack([q_mean[i, max_q_mean_idx[i]] for i in range(q_mean.shape[0])])
        else:
            raise ValueError("Invalid filter type!!!")

        self.v[0].info["q_of_chosen_action"].append(torch.mean(max_q_mean_val).item())
        return max_q_mean_idx

    def sample(self, states, sample_per_state=16, diffusion_steps=15):
        self.eval()
        num_states = states.shape[0]
        with torch.no_grad():
            states = torch.FloatTensor(states).to(self.device)
            states = torch.repeat_interleave(states, sample_per_state, dim=0)
            self.condition = states
            results = self.dpm_wrapper_sample(self.output_dim, batch_size=states.shape[0], steps=diffusion_steps, order=2)
            actions = results[:, :].reshape(num_states, sample_per_state, self.output_dim).copy()
            self.condition = None
        self.train()
        return actions


class ScoreNet(ScoreBase):
    def __init__(self, input_dim, output_dim, marginal_prob_std, embed_dim=32, **kwargs):
        super().__init__(input_dim, output_dim, marginal_prob_std, embed_dim, **kwargs)
        # The swish activation function
        self.act = lambda x: x * torch.sigmoid(x)
        self.pre_sort_condition = nn.Sequential(Dense(input_dim-output_dim, 32), SiLU())
        self.sort_t = nn.Sequential(
                        nn.Linear(64, 128),                        
                        SiLU(),
                        nn.Linear(128, 128),
                    )
        self.down_block1 = Residual_Block(output_dim, 512)
        self.down_block2 = Residual_Block(512, 256)
        self.down_block3 = Residual_Block(256, 128)
        self.middle1 = Residual_Block(128, 128)
        self.up_block3 = Residual_Block(256, 256)
        self.up_block2 = Residual_Block(512, 512)
        self.last = nn.Linear(1024, output_dim)
        
    def forward(self, x, t, condition=None):
        embed = self.embed(t)
        
        if condition is not None:
            embed = torch.cat([self.pre_sort_condition(condition), embed], dim=-1)
        else:
            if self.condition.shape[0] == x.shape[0]:
                condition = self.condition
            elif self.condition.shape[0] == 1:
                condition = torch.cat([self.condition]*x.shape[0])
            else:
                assert False
            embed = torch.cat([self.pre_sort_condition(condition), embed], dim=-1)
        embed = self.sort_t(embed)
        d1 = self.down_block1(x, embed)
        d2 = self.down_block2(d1, embed)
        d3 = self.down_block3(d2, embed)
        u3 = self.middle1(d3, embed)
        u2 = self.up_block3(torch.cat([d3, u3], dim=-1), embed)
        u1 = self.up_block2(torch.cat([d2, u2], dim=-1), embed)
        u0 = torch.cat([d1, u1], dim=-1)
        h = self.last(u0)
        self.h = h
        # Normalize output
        return h / self.marginal_prob_std(t)[1][..., None]