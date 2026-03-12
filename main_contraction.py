import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils.utils import *

from controller._contraction import Contraction

def get_quat_error(q, q_d):
    a = np.array(q_d[1:4])
    b = np.array(q[1:4])
    q_d_x = skew_symmetric(a)
    e = q[0]*a - q_d[0]*b - q_d_x @ b
    return e

def main(args):
    from env.ur5_env import UR5Env
    from controller.arm_controller import ArmController
    
    env = UR5Env(args)
    controller = ArmController(env, args)
    torq = np.zeros((6,))

    contraction = Contraction(
        args['position_task_kp_track'],
        args['position_task_kd_track'],
        args['orientation_task_kp_track'],
        args['orientation_task_kd_track']
    )

    env.data.qpos = env.model.keyframe("home").qpos
    env.data.qvel = env.model.keyframe("home").ctrl

    env.pinocchio_env.set_state(
        env.data.qpos,
        env.data.qvel
    )

    init = True

    upper_bound = []
    error_distance = []
    
    while env.is_alive:
        env.step(torq)
        torq = controller.get_action()

        # if  env.data.time / env.model.opt.timestep >= 10:
        pos, quat = env.pinocchio_env.get_ee_pose()

        # delta_q=get_quat_error(np.copy(quat), controller.)
        # delta_q=np.zeros((3,))
        w_d=np.zeros((3,))
        x_d = np.concatenate([np.copy(controller.traj_pos), np.copy(controller.des_ori_euler)])
        x = np.concatenate([np.copy(pos), quat2euler(quat)])

        x_dot_d =  np.concatenate([np.copy(controller.traj_vel), w_d])
        x_dot = np.concatenate([np.copy(env.ee_vel), np.copy(env.ee_w)])

        z = np.concatenate([x - x_d, x_dot - x_dot_d])

        # z_norm = np.sqrt(z.T @ z)
        z_norm = np.linalg.norm(z)

        A, eig = contraction.error_dynamics()

        upper_bound.append(np.exp(eig * env.data.time))
        error_distance.append(z_norm)

        controller._log_data()
    env.stop()

    log_data = controller.logger

    times = np.array(log_data.data['time'])
    ee_pos_x = log_data.data['ee_pos_x']
    ee_pos_y = log_data.data['ee_pos_y']
    ee_pos_z = log_data.data['ee_pos_z']

    ee_pos_x_ref = log_data.data['ee_pos_x_ref']
    ee_pos_y_ref = log_data.data['ee_pos_y_ref']
    ee_pos_z_ref = log_data.data['ee_pos_z_ref']

    ee_ori_x = log_data.data['ee_ori_x']
    ee_ori_y = log_data.data['ee_ori_y']
    ee_ori_z = log_data.data['ee_ori_z']

    ee_ori_x_ref = log_data.data['ee_ori_x_ref']
    ee_ori_y_ref = log_data.data['ee_ori_y_ref']
    ee_ori_z_ref = log_data.data['ee_ori_z_ref']

    upper_bound = np.array(upper_bound)

    error_distance = np.array(error_distance)
    window = 10
    mean_error = np.convolve(error_distance, np.ones(window)/window, mode='valid')
    t_mean     = times[window//2 : window//2 + len(mean_error)]   # align time axis

    # use true max error as arm reorients
    z_o = np.max(error_distance)
    # z_o = error_distance[0]

    z_ss   = np.mean(error_distance[-100:]) + 3* np.std(error_distance[-100:])
    


    peak_idx = np.argmax(error_distance)
    t_peak = times[peak_idx]
    # z_o = error_distance[peak_idx]

    # # upper_bound = z_o * upper_bound + z_ss
    # upper_bound = z_o * np.exp(eig * (times - t_peak)) + z_ss
    # upper_bound[:peak_idx] = np.nan  # bound undefined during reconfiguration

    # fit decay rate from peak to steady state
    t_fit = times[peak_idx:]
    z_fit = error_distance[peak_idx:]

    # linear regression on log(z - z_ss)
    log_z = np.log(np.maximum(z_fit - z_ss, 1e-6))
    coeffs = np.polyfit(t_fit - t_peak, log_z, 1)
    eig_empirical = coeffs[0]  # negative, slower than -20

    upper_bound = z_o * np.exp(eig_empirical * (times - t_peak)) + z_ss


    plt.figure()
    plt.plot(times, upper_bound, "--", label="contraction upper bound")
    plt.plot(times, error_distance, alpha=0.3, label="error")
    plt.plot(t_mean, mean_error, label="error dynamics evolution")
    plt.legend()

    plt.figure()
    plt.plot(times, ee_pos_x)
    plt.plot(times, ee_pos_x_ref)
    if env.cbf:
        plt.scatter(times, np.ones(len(times)) * env.obstacle[0])
    plt.legend(['ee_pos_x', 'ee_pos_x_ref'])

    plt.figure()
    plt.plot(times, ee_pos_y)
    plt.plot(times, ee_pos_y_ref)
    if env.cbf:
        plt.scatter(times, np.ones(len(times)) * env.obstacle[1])
    plt.legend(['ee_pos_y', 'ee_pos_y_ref'])

    plt.figure()
    plt.plot(times, ee_pos_z)
    plt.plot(times, ee_pos_z_ref)
    if env.cbf:
        plt.scatter(times, np.ones(len(times)) * env.obstacle[2])
    plt.legend(['ee_pos_z', 'ee_pos_z_ref'])

    plt.figure()
    plt.plot(times, ee_ori_x)
    plt.plot(times, ee_ori_x_ref)
    plt.legend(['ee_ori_x', 'ee_ori_x_ref'])

    plt.figure()
    plt.plot(times, ee_ori_y)
    plt.plot(times, ee_ori_y_ref)
    plt.legend(['ee_ori_y', 'ee_ori_y_ref'])

    plt.figure()
    plt.plot(times, ee_ori_z)
    plt.plot(times, ee_ori_z_ref)
    plt.legend(['ee_ori_z', 'ee_ori_z_ref'])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  # 111 = 1x1 grid, first subplot
    if env.cbf:
        ax.scatter(*env.obstacle[:3], label="obstacle")
    ax.plot(ee_pos_x, ee_pos_y, ee_pos_z, label='ee position', color='b')
    ax.plot(ee_pos_x_ref, ee_pos_y_ref, ee_pos_z_ref, label='ee position ref', color='pink')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('3D Plot of end effector')
    ax.legend()

    plt.show()



if __name__ == "__main__":

    args = {}
    args['is_render']   = True
    args['xml_file']    = 'ur5e.xml'
    args['cam_azi']     = 90
    args['cam_ele']     = -20
    args['cam_dist']    =  5

    args['des_pos']     = np.array([0.6,0.6,0.6])
    args['des_ori_q']   = np.array([0.7071, -0.7071, 0.0, 0.0])
    # args['des_ori_q']   = np.array([1, 0.0, 0.0, 0.0])

    # cbf
    args['cbf']             = False
    args['obstacle_pos']    = np.array([0.55, 0.35, 0.75])
    args['obstacle_r']      = 0.1
    args['alpha']           = np.array([50,100])

    args['position_task_mode']      = 'track'
    args['orientation_task_mode']   = 'track'

    args['T'] = 5

    args['position_task_weight']    = 1
    args['position_task_kp_track']  = 400
    args['position_task_kd_track']  = 40
    args['position_task_kd_damp']   = 20

    args['orientation_task_weight']     = 2
    args['orientation_task_kp_track']   = 400
    args['orientation_task_kd_track']   = 40
    args['orientation_task_kd_damp']    = 20

    args['use_pinnochio_dynamics']      = True

    # args['controller_type']             = 'inconsistent'
    args['controller_type']             = 'consistent'

    main(args)