import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Make project root importable (for env, utils, etc.)
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from utils.utils import quat2euler
import train  # imports UR5TrackingEnv, Agent, PPOConfig


def main():
    args = {}
    args["is_render"] = True
    args["xml_file"] = "ur5e.xml"
    args["cam_azi"] = 90
    args["cam_ele"] = -20
    args["cam_dist"] = 5.0

    args["des_pos"] = np.array([0.6, 0.4, 0.5])
    args["des_ori_q"] = np.array([1.0, 0.0, 0.0, 0.0])

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

    # args["initial_ee_pos_range"] = np.array([0.03, 0.03, 0.03])  # ±3 cm; use zeros to disable
    args["initial_ee_pos_range"] = np.array([0.0, 0.0, 0.0])  # ±3 cm; use zeros to disable

    cfg = train.PPOConfig()
    model_path = "runs/ur5_ppo_contraction_step20021248.pt"

    device = torch.device(cfg.device)

    # ---- Create env first so you can pass action bounds to Agent ----
    env = train.UR5TrackingEnv(args)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    agent = train.Agent(
        obs_dim,
        act_dim,
        action_low=env.action_space.low,
        action_high=env.action_space.high,
    ).to(device)

    state_dict = torch.load(model_path, map_location=device)
    # Support both plain agent weights and full training checkpoints
    if isinstance(state_dict, dict) and "agent" in state_dict:
        state_dict = state_dict["agent"]
    agent.load_state_dict(state_dict)
    agent.eval()

    # ---- Logging buffers ----
    times = []
    ee_pos_x = []
    ee_pos_y = []
    ee_pos_z = []

    ee_pos_x_ref = []
    ee_pos_y_ref = []
    ee_pos_z_ref = []

    ee_ori_x = []
    ee_ori_y = []
    ee_ori_z = []

    ee_ori_x_ref = []
    ee_ori_y_ref = []
    ee_ori_z_ref = []

    des_ori_euler = quat2euler(args["des_ori_q"])

    # ---- Rollout one episode ----
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

        t = env.time
        times.append(t)

        pos = env.env.ee_pos
        ori_euler = env.env.ee_euler

        ee_pos_x.append(pos[0])
        ee_pos_y.append(pos[1])
        ee_pos_z.append(pos[2])

        ee_ori_x.append(ori_euler[0])
        ee_ori_y.append(ori_euler[1])
        ee_ori_z.append(ori_euler[2])

        ref_pos = env.traj_pos

        ee_pos_x_ref.append(ref_pos[0])
        ee_pos_y_ref.append(ref_pos[1])
        ee_pos_z_ref.append(ref_pos[2])

        ee_ori_x_ref.append(des_ori_euler[0])
        ee_ori_y_ref.append(des_ori_euler[1])
        ee_ori_z_ref.append(des_ori_euler[2])

    env.close()

    times = np.array(times)

    ee_pos_x = np.array(ee_pos_x)
    ee_pos_y = np.array(ee_pos_y)
    ee_pos_z = np.array(ee_pos_z)

    ee_pos_x_ref = np.array(ee_pos_x_ref)
    ee_pos_y_ref = np.array(ee_pos_y_ref)
    ee_pos_z_ref = np.array(ee_pos_z_ref)

    ee_ori_x = np.array(ee_ori_x)
    ee_ori_y = np.array(ee_ori_y)
    ee_ori_z = np.array(ee_ori_z)

    ee_ori_x_ref = np.array(ee_ori_x_ref)
    ee_ori_y_ref = np.array(ee_ori_y_ref)
    ee_ori_z_ref = np.array(ee_ori_z_ref)

    # ---- Plots: position ----
    plt.figure()
    plt.plot(times, ee_pos_x)
    plt.plot(times, ee_pos_x_ref, "--")
    plt.legend(["ee_pos_x", "ee_pos_x_ref"])
    plt.xlabel("time [s]")
    plt.ylabel("x [m]")
    plt.title("End-effector X position")

    plt.figure()
    plt.plot(times, ee_pos_y)
    plt.plot(times, ee_pos_y_ref, "--")
    plt.legend(["ee_pos_y", "ee_pos_y_ref"])
    plt.xlabel("time [s]")
    plt.ylabel("y [m]")
    plt.title("End-effector Y position")

    plt.figure()
    plt.plot(times, ee_pos_z)
    plt.plot(times, ee_pos_z_ref, "--")
    plt.legend(["ee_pos_z", "ee_pos_z_ref"])
    plt.xlabel("time [s]")
    plt.ylabel("z [m]")
    plt.title("End-effector Z position")

    # ---- Plots: orientation ----
    plt.figure()
    plt.plot(times, ee_ori_x)
    plt.plot(times, ee_ori_x_ref, "--")
    plt.legend(["ee_ori_x", "ee_ori_x_ref"])
    plt.xlabel("time [s]")
    plt.ylabel("roll [rad]")
    plt.title("End-effector roll")

    plt.figure()
    plt.plot(times, ee_ori_y)
    plt.plot(times, ee_ori_y_ref, "--")
    plt.legend(["ee_ori_y", "ee_ori_y_ref"])
    plt.xlabel("time [s]")
    plt.ylabel("pitch [rad]")
    plt.title("End-effector pitch")

    plt.figure()
    plt.plot(times, ee_ori_z)
    plt.plot(times, ee_ori_z_ref, "--")
    plt.legend(["ee_ori_z", "ee_ori_z_ref"])
    plt.xlabel("time [s]")
    plt.ylabel("yaw [rad]")
    plt.title("End-effector yaw")

    # ---- 3D trajectory ----
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(ee_pos_x, ee_pos_y, ee_pos_z, label="ee position")
    ax.plot(ee_pos_x_ref, ee_pos_y_ref, ee_pos_z_ref, "--", label="ee position ref")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D End-effector trajectory")
    ax.legend()

    plt.show()


if __name__ == "__main__":
    main()