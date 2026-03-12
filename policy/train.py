import os
import time
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from env.ur5_env import UR5Env
from utils.trajectory_generator import TrajectoryGenerator
from utils.utils import skew_symmetric, quat2euler
from controller.contraction import Contraction


# ==========================
#  UR5 Gym env with CLF-style internal trajectory
# ==========================

class UR5TrackingEnv(gym.Env):
    """
    Observation: [current_state(13), tracking_error(12)] -> R^25
      current: ee_pos(3), ee_quat(4), ee_vel(3), ee_w(3)
      error:   pos_err(3), quat_err(3), vel_err(3), w_err(3)
    Action:     joint torques (6-dim box)
    Reference:  trajectory from home -> des_pos over T (same in world); policy sees ref at each step.
    """
    metadata = {"render_modes": ["human"]} 

    def __init__(self, ur5_args: dict):
        super().__init__()
        self.args = ur5_args

        # Create underlying Mujoco env
        self.env = UR5Env(ur5_args)
        self.dt = self.env.model.opt.timestep

        # Trajectory generator (same as CLF controllers)
        self.des_pos = ur5_args["des_pos"]
        self.des_ori_q = ur5_args["des_ori_q"]
        self.T = ur5_args["T"]
        self.traj_handler = TrajectoryGenerator(self.dt)
        self.time = 0.0

        # Contraction reward shaping (optional)
        self._contraction_reward_w = float(ur5_args.get("contraction_reward_weight", 1.0))
        self._contraction_reward_beta = float(ur5_args.get("contraction_reward_beta", 1.0))
        self._contraction = Contraction(
            Kp_pos=ur5_args["position_task_kp_track"],
            Kd_pos=ur5_args["position_task_kd_track"],
            Kp_ori=ur5_args["orientation_task_kp_track"],
            Kd_ori=ur5_args["orientation_task_kd_track"],
        )
        self._des_ori_euler = quat2euler(self.des_ori_q)

        # Initial EE position randomization: half-extents (m) for uniform sampling around home.
        # Robot starts at home + offset; trajectory is always home -> des_pos (unchanged).
        self.initial_ee_pos_range = np.asarray(
            ur5_args.get("initial_ee_pos_range", np.zeros(3)), dtype=np.float64
        )
        self._ik_step = 0.5
        self._ik_iters = 25

        # Store last ref state for logging / reward
        self.traj_pos = self.env.ee_pos.copy()
        self.traj_vel = np.zeros(3)
        self.traj_acc = np.zeros(3)

        # Observation: current state (13) + tracking error (12) = 25-dim
        # [ee_pos(3), ee_q(4), ee_vel(3), ee_w(3), pos_err(3), quat_err(3), vel_err(3), w_err(3)]
        obs_dim = 25
        high_obs = np.ones(obs_dim, dtype=np.float32) * np.inf
        self.observation_space = gym.spaces.Box(
            low=-high_obs, high=high_obs, dtype=np.float32
        )

        # Action: torques, use Mujoco actuator ctrlrange if available
        if hasattr(self.env.model, "actuator_ctrlrange"):
            ctrl_range = self.env.model.actuator_ctrlrange.copy()
            low = ctrl_range[:, 0]
            high = ctrl_range[:, 1]
        else:
            # Fallback range if not defined
            low = -np.ones(6) * 15.0
            high = np.ones(6) * 15.0

        self.action_space = gym.spaces.Box(
            low=low.astype(np.float32),
            high=high.astype(np.float32),
            dtype=np.float32,
        )

    def _get_obs(self) -> np.ndarray:
        # Current state
        ee_pos = self.env.ee_pos          # (3,)
        ee_q = self.env.ee_q              # (4,)
        ee_vel = self.env.ee_vel          # (3,)
        ee_w = self.env.ee_w              # (3,)

        # Tracking errors (policy sees what to correct)
        ref_pos = self.traj_pos
        ref_vel = self.traj_vel
        ref_q = self.des_ori_q
        ref_w = np.zeros(3, dtype=np.float32)

        pos_err = ee_pos - ref_pos

        # Quaternion error (same as task_space_objective.EEOrientationTask.get_quat_error)
        a = np.array(ref_q[1:4])
        b = np.array(ee_q[1:4])
        q_d_x = skew_symmetric(a)
        quat_err = ee_q[0] * a - ref_q[0] * b - q_d_x @ b  # (3,)

        # vel_err = ee_vel - ref_vel
        # w_err = ee_w - ref_w

        obs = np.concatenate([
            ee_pos, ee_q, ee_vel, ee_w,
            pos_err, quat_err, ref_vel, ref_w,
        ]).astype(np.float32)
        return obs

    def _tracking_reward(self, obs: np.ndarray, action: np.ndarray) -> float:
        # obs layout: [current(13), ref(13)]; use ref from env for consistency
        ref_pos = self.traj_pos
        ref_vel = self.traj_vel
        ref_ori_q = self.des_ori_q
        ref_w = np.zeros(3)

        pos_err = obs[0:3] - ref_pos
        q = obs[3:7]
        q_ref = ref_ori_q
        # Quaternion error (same as task_space_objective.EEOrientationTask.get_quat_error)
        a = np.array(q_ref[1:4])
        b = np.array(q[1:4])
        q_d_x = skew_symmetric(a)
        quat_err = q[0] * a - q_ref[0] * b - q_d_x @ b  # 3D vector
        vel_err = obs[7:10] - ref_vel
        w_err = obs[10:13] - ref_w

        w_pos = 10.0
        w_quat = 2.0
        w_vel = 0.15
        w_w = 0.1
        w_u = 0.01

        scale = 3.0
        cost = (
            w_pos * np.sum(pos_err ** 2)
            + w_quat * np.sum(quat_err ** 2)
            + w_vel * np.sum(vel_err ** 2)
            + w_w * np.sum(w_err ** 2)
            + w_u * np.sum(action ** 2)
        )

        return np.exp(-cost / scale)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # Reset underlying mujoco state (like main_clf)
        self.env.data.qpos[:] = self.env.model.keyframe("home").qpos.copy()
        self.env.data.qvel[:] = 0.0

        import mujoco as mj
        mj.mj_forward(self.env.model, self.env.data)
        self.env.update_robot_states()

        # Trajectory is always from nominal home to des_pos (same every time)
        home_ee_pos = self.env.ee_pos.copy()
        self.traj_handler.reset_trajectory(
            home_ee_pos,
            self.des_pos,
            np.zeros(3),
            np.zeros(3),
            self.T,
        )
        self.time = 0.0

        # Optionally randomize where the robot starts (same trajectory, different initial pose)
        if np.any(self.initial_ee_pos_range > 0):
            target_ee_pos = home_ee_pos + np.random.uniform(
                -self.initial_ee_pos_range, self.initial_ee_pos_range
            )
            self._solve_ik_position(target_ee_pos)

        # First reference point (at t=0 ref is still home_ee_pos; robot may be offset)
        self.traj_pos, self.traj_vel, self.traj_acc = self.traj_handler.get_trajectory()

        obs = self._get_obs()
        info = {}
        return obs, info

    def _clamp_qpos_to_limits(self):
        """Clamp current qpos to model joint limits (first 6 joints = arm)."""
        import mujoco as mj
        model = self.env.model
        data = self.env.data
        for j in range(min(6, model.njnt)):
            adr = model.jnt_qposadr[j]
            low, high = model.jnt_range[j, 0], model.jnt_range[j, 1]
            if np.isfinite(low) and np.isfinite(high):
                data.qpos[adr] = np.clip(data.qpos[adr], low, high)

    def _solve_ik_position(self, target_ee_pos: np.ndarray):
        """Move robot so EE position reaches target_ee_pos (position-only IK)."""
        import mujoco as mj
        for _ in range(self._ik_iters):
            delta = target_ee_pos - self.env.ee_pos
            if np.linalg.norm(delta) < 1e-4:
                break
            J = self.env.jacp  # (3, nv)
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-2 * np.eye(3), delta)
            self.env.data.qpos[:6] += self._ik_step * dq
            self._clamp_qpos_to_limits()
            mj.mj_forward(self.env.model, self.env.data)
            self.env.update_robot_states()

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.env.step(action)

        # Update reference
        self.traj_pos, self.traj_vel, self.traj_acc = self.traj_handler.get_trajectory()

        obs = self._get_obs()
        # reward = self._tracking_reward(obs, action)

        reward = 0.0
        # Add contraction reward (no termination logic added here)
        if self._contraction_reward_w > 0.0:
            x = np.concatenate([self.env.ee_pos, quat2euler(self.env.ee_q)])
            x_d = np.concatenate([self.traj_pos, self._des_ori_euler])
            x_dot = np.concatenate([self.env.ee_vel, self.env.ee_w])
            x_dot_d = np.concatenate([self.traj_vel, np.zeros(3)])
            z_norm, _ = self._contraction.get_upper_bound(self.time, x, x_d, x_dot, x_dot_d)
            reward += self._contraction_reward_w * float(np.exp(-self._contraction_reward_beta * z_norm))
        # print(self._contraction_reward_w * float(np.exp(-self._contraction_reward_beta * z_norm)))
        self.time += self.dt
        # reward += 1.0
        terminated = False
        truncated = self.time >= self.T - self.dt  # episode ends when trajectory duration elapsed

        # early termination if EE strays too far from reference position
        pos_err = np.linalg.norm(self.env.ee_pos - self.traj_pos)
        if pos_err > 0.05:
            terminated = True
            reward -= 10.0

        if not self.env.is_alive:
            terminated = True
            reward -= 10.0

        if truncated and not terminated:
            reward += 10.0

        info = {"pos_err": pos_err}
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.env.is_render:
            self.env.render()

    def close(self):
        self.env.stop()


# ==========================
#  PPO agent (based on policy/ppo.py)
# ==========================

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, action_low, action_high):
        super().__init__()

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 1), std=1.0),
        )

        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, act_dim), std=0.01),
        )

        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

        action_low = torch.as_tensor(action_low, dtype=torch.float32)
        action_high = torch.as_tensor(action_high, dtype=torch.float32)

        # Buffers move with .to(device)
        self.register_buffer("action_low", action_low)
        self.register_buffer("action_high", action_high)
        self.register_buffer("action_scale", 0.5 * (action_high - action_low))
        self.register_buffer("action_bias", 0.5 * (action_high + action_low))

    def get_value(self, x):
        return self.critic(x)

    def _normal_dist(self, x):
        mean = self.actor_mean(x)
        logstd = self.actor_logstd.expand_as(mean).clamp(-5.0, 2.0)
        std = torch.exp(logstd)
        return Normal(mean, std), mean

    def _squash_action(self, u):
        # u in R^n -> a in [low, high]
        squashed = torch.tanh(u)
        action = squashed * self.action_scale + self.action_bias
        return action, squashed

    def _unsquash_action(self, action):
        # map action in [low, high] back to pre-scaled tanh range (-1, 1)
        eps = 1e-6
        y = (action - self.action_bias) / (self.action_scale + eps)
        y = torch.clamp(y, -1.0 + eps, 1.0 - eps)
        u = 0.5 * (torch.log1p(y) - torch.log1p(-y))  # atanh(y)
        return u, y

    def get_action_and_value(self, x, action=None):
        dist, mean = self._normal_dist(x)

        if action is None:
            u = dist.rsample()  # reparameterized sample
            action, y = self._squash_action(u)
        else:
            u, y = self._unsquash_action(action)

        # Log prob with tanh squash correction and affine scaling correction
        # a = scale * tanh(u) + bias
        # log|da/du| = log(scale) + log(1 - tanh(u)^2)
        eps = 1e-6
        log_prob_u = dist.log_prob(u)
        log_det_jacobian = torch.log(self.action_scale + eps) + torch.log(1 - y.pow(2) + eps)
        log_prob = (log_prob_u - log_det_jacobian).sum(dim=-1)

        entropy = dist.entropy().sum(dim=-1)  # approximation; common in PPO
        value = self.critic(x)

        return action, log_prob, entropy, value

    def get_deterministic_action(self, x):
        mean = self.actor_mean(x)
        action, _ = self._squash_action(mean)
        return action


# ==========================
#  PPO hyperparameters
# ==========================

@dataclass
class PPOConfig:
    total_timesteps: int = 50_000_000
    num_envs: int = 64          # number of parallel environments
    num_steps: int = 512        # rollout length per env
    gamma: float = 0.995
    gae_lambda: float = 0.95
    learning_rate: float = 3e-4
    num_minibatches: int = 128
    update_epochs: int = 5
    clip_coef: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None
    seed: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_dir: str = "runs"


def make_env(ur5_args: dict, idx: int, seed: int):
    def thunk():
        env_args = dict(ur5_args)
        env_args["is_render"] = False
        env = UR5TrackingEnv(env_args)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        # env = gym.wrappers.ClipAction(env)
        return env

    return thunk


# ==========================
#  Training loop
# ==========================

def train(ur5_args: dict, cfg: PPOConfig, save_path: str, load_path: str | None = None):
    next_ckpt = 10_000_000
    run_name = f"ur5_track_ppo__{int(time.time())}"
    writer = SummaryWriter(os.path.join(cfg.log_dir, run_name))

    # seeding
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device(cfg.device)
    print(f"Using device: {device}")
    # vectorized envs
    envs = gym.vector.SyncVectorEnv(
        [make_env(ur5_args, i, cfg.seed + i) for i in range(cfg.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box)

    obs_shape = envs.single_observation_space.shape
    act_shape = envs.single_action_space.shape

    agent = Agent(
        obs_dim=int(np.array(obs_shape).prod()),
        act_dim=int(np.array(act_shape).prod()),
        action_low=envs.single_action_space.low,
        action_high=envs.single_action_space.high,
    ).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=cfg.learning_rate, eps=1e-5)

    # Resume from checkpoint (optional)
    global_step = 0
    start_iteration = 1
    if load_path is not None and os.path.isfile(load_path):
        ckpt = torch.load(load_path, map_location=device)
        if isinstance(ckpt, dict) and "agent" in ckpt:
            agent.load_state_dict(ckpt["agent"])
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            global_step = ckpt.get("global_step", 0)
            next_ckpt = ckpt.get("next_ckpt", ((global_step // 1_000_000) + 1) * 1_000_000)
            start_iteration = (global_step // (cfg.num_steps * cfg.num_envs)) + 1
            print(f"Resumed from {load_path}: global_step={global_step}, next_ckpt={next_ckpt}")
        else:
            agent.load_state_dict(ckpt)
            print(f"Loaded agent weights from {load_path} (no optimizer/step; training from step 0)")

    num_envs = cfg.num_envs
    batch_size = cfg.num_steps * num_envs
    minibatch_size = batch_size // cfg.num_minibatches
    num_iterations = cfg.total_timesteps // batch_size

    obs = torch.zeros((cfg.num_steps, num_envs) + obs_shape, device=device)
    actions = torch.zeros((cfg.num_steps, num_envs) + act_shape, device=device)
    logprobs = torch.zeros((cfg.num_steps, num_envs), device=device)
    rewards = torch.zeros((cfg.num_steps, num_envs), device=device)
    dones = torch.zeros((cfg.num_steps, num_envs), device=device)
    values = torch.zeros((cfg.num_steps, num_envs), device=device)

    # manual per-env episode stats
    ep_returns = np.zeros(num_envs, dtype=np.float32)
    ep_lengths = np.zeros(num_envs, dtype=np.int32)

    run_start_global_step = global_step
    run_start_time = time.time()

    next_obs, _ = envs.reset(seed=cfg.seed)
    next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
    next_done = torch.zeros(num_envs, device=device)

    for iteration in range(start_iteration, num_iterations + 1):
        # Rollout
        for step in range(cfg.num_steps):
            global_step += num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                flat_obs = next_obs.view(num_envs, -1)
                action, logprob, _, value = agent.get_action_and_value(flat_obs)
                values[step] = value.view(-1)
            actions[step] = action.view((num_envs,) + act_shape)
            logprobs[step] = logprob

            next_obs_np, reward, terminated, truncated, infos = envs.step(
                action.cpu().numpy()
            )
            next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
            done = np.logical_or(terminated, truncated)
            next_done = torch.tensor(done, dtype=torch.float32, device=device)
            rewards[step] = torch.tensor(reward, dtype=torch.float32, device=device)
            # manual episodic stats logging per env
            ep_returns += reward
            ep_lengths += 1
            for i, d in enumerate(done):
                if d:
                    print(f"global_step={global_step}, episodic_return={ep_returns[i]}, episodic_length={ep_lengths[i]}")
                    writer.add_scalar("charts/episodic_return", ep_returns[i], global_step)
                    writer.add_scalar("charts/episodic_length", ep_lengths[i], global_step)
                    ep_returns[i] = 0.0
                    ep_lengths[i] = 0
        # GAE / returns
        with torch.no_grad():
            flat_next_obs = next_obs.view(num_envs, -1)
            next_value = agent.get_value(flat_next_obs).view(1, -1)
            advantages = torch.zeros_like(rewards, device=device)
            lastgaelam = 0
            for t in reversed(range(cfg.num_steps)):
                if t == cfg.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + cfg.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = (
                    delta + cfg.gamma * cfg.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        # Flatten batch
        b_obs = obs.view((-1,) + obs_shape)
        b_obs_flat = b_obs.view(batch_size, -1)
        b_actions = actions.view((-1,) + act_shape)
        b_logprobs = logprobs.view(-1)
        b_advantages = advantages.view(-1)
        b_returns = returns.view(-1)
        b_values = values.view(-1)

        # PPO update
        b_inds = np.arange(batch_size)
        clipfracs = []
        pg_losses, v_losses, entropies, approx_kls, old_approx_kls = [], [], [], [], []

        for epoch in range(cfg.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                mb_obs = b_obs_flat[mb_inds]
                mb_actions = b_actions.view(batch_size, -1)[mb_inds]
                mb_logprobs_old = b_logprobs[mb_inds]
                mb_advantages = b_advantages[mb_inds]
                mb_returns = b_returns[mb_inds]
                mb_values = b_values[mb_inds]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    mb_obs, mb_actions
                )
                logratio = newlogprob - mb_logprobs_old
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(
                        ((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item()
                    )

                mb_adv = mb_advantages
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # policy loss
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(
                    ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # value loss
                newvalue = newvalue.view(-1)
                v_loss_unclipped = (newvalue - mb_returns) ** 2
                v_clipped = mb_values + torch.clamp(
                    newvalue - mb_values, -cfg.clip_coef, cfg.clip_coef
                )
                v_loss_clipped = (v_clipped - mb_returns) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - cfg.ent_coef * entropy_loss + cfg.vf_coef * v_loss

                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                entropies.append(entropy_loss.item())
                approx_kls.append(approx_kl.item())
                old_approx_kls.append(old_approx_kl.item())

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                optimizer.step()

            if cfg.target_kl is not None and approx_kl > cfg.target_kl:
                break

        # diagnostics
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        sps = int((global_step - run_start_global_step) / (time.time() - run_start_time))
        print(f"Iteration {iteration}/{num_iterations}, global_step={global_step}, SPS={sps}")

        writer.add_scalar("charts/SPS", sps, global_step)
        writer.add_scalar("losses/value_loss", np.mean(v_losses), global_step)
        writer.add_scalar("losses/policy_loss", np.mean(pg_losses), global_step)
        writer.add_scalar("losses/entropy", np.mean(entropies), global_step)
        writer.add_scalar("losses/old_approx_kl", np.mean(old_approx_kls), global_step)
        writer.add_scalar("losses/approx_kl", np.mean(approx_kls), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)

        # periodic checkpoints every 200k environment steps (full ckpt for resume)
        if global_step >= next_ckpt:
            ckpt_dir = os.path.dirname(save_path)
            os.makedirs(ckpt_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(save_path))[0]
            ckpt_path = os.path.join(ckpt_dir, f"{base}_step{global_step}.pt")
            torch.save({
                "agent": agent.state_dict(),
                "optimizer": optimizer.state_dict(),
                "global_step": global_step,
                "next_ckpt": next_ckpt + 10_000_000,
            }, ckpt_path)
            next_ckpt += 10_000_000
            print(f"Checkpoint saved to {ckpt_path}")

    # Save trained model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(agent.state_dict(), save_path)
    print(f"Saved trained policy to {save_path}")

    envs.close()
    writer.close()
    return agent


# ==========================
#  Deployment / evaluation
# ==========================

def deploy(ur5_args: dict, model_path: str, cfg: PPOConfig):
    env = UR5TrackingEnv(ur5_args)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    device = torch.device(cfg.device)
    agent = Agent(
        obs_dim,
        act_dim,
        action_low=env.action_space.low,
        action_high=env.action_space.high,
    ).to(device)

    state_dict = torch.load(model_path, map_location=device)
    agent.load_state_dict(state_dict)
    agent.eval()

    obs_np, _ = env.reset()
    done = False

    while not done and env.env.is_alive:
        obs = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            obs_flat = obs.view(1, -1)
            action = agent.get_deterministic_action(obs_flat)

        action_np = action.cpu().numpy()[0]
        obs_np, reward, terminated, truncated, _ = env.step(action_np)
        done = terminated or truncated
        env.render()

    env.close()


# ==========================
#  Example usage
# ==========================

if __name__ == "__main__":
    args = {}
    args["is_render"] = False          # False for training, True for deploy
    args["xml_file"] = "ur5e.xml"
    args["cam_azi"] = 90
    args["cam_ele"] = -20
    args["cam_dist"] = 5.0

    # Same fields `UR5Env` / controllers expect
    args["des_pos"] = np.array([0.6, 0.4, 0.5])          # final EE position
    args["des_ori_q"] = np.array([1.0, 0.0, 0.0, 0.0])   # fixed orientation
    # Robot starts at home + uniform offset (m); trajectory unchanged (home -> des_pos)
    args["initial_ee_pos_range"] = np.array([0.03, 0.03, 0.03])  # ±3 cm; use zeros to disable

    args["cbf"] = False
    args["obstacle_pos"] = np.array([0.55, 0.35, 0.75])
    args["obstacle_r"] = 0.1
    args["alpha"] = np.array([50, 100])

    args["position_task_mode"] = "track"
    args["orientation_task_mode"] = "track"

    args["T"] = 5.0

    args["position_task_weight"] = 1
    args["position_task_kp_track"] = 600
    args["position_task_kd_track"] = 60
    args["position_task_kd_damp"] = 20

    args["orientation_task_weight"] = 2
    args["orientation_task_kp_track"] = 600
    args["orientation_task_kd_track"] = 60
    args["orientation_task_kd_damp"] = 20

    args["use_pinnochio_dynamics"] = True
    args["controller_type"] = "consistent"

    cfg = PPOConfig()
    # model_path = "runs/ur5_ppo.pt"
    model_path = "runs/ur5_ppo_contraction.pt"
    resume_from = None #"runs/ur5_ppo_10mil.pt"

    # Train (optionally resume from checkpoint)
    trained_agent = train(args, cfg, model_path, load_path=resume_from)

    # Deploy with rendering
    args["is_render"] = True
    deploy(args, model_path, cfg)