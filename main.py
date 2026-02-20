import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def main(args):
    from env.ur5_env import UR5Env
    from controller.arm_controller import ArmController
    
    env = UR5Env(args)
    controller = ArmController(env, args)
    torq = np.zeros((6,))

    env.data.qpos = env.model.keyframe("home").qpos
    

    while env.is_alive:
        env.step(torq)
        torq = controller.get_action()
        controller._log_data()
    env.stop()

    log_data = controller.logger

    times = log_data.data['time']

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
    args['is_render']  = True
    args['xml_file']   = 'ur5e.xml'
    args['cam_azi'] = 90
    args['cam_ele'] = -20
    args['cam_dist'] =  5

    args['des_pos'] = np.array([0.6,0.6,0.6])
    args['des_ori_q'] = np.array([1, 0.0, 0.0, 0])

    # cbf
    args['cbf']             = True
    args['obstacle_pos']    = np.array([0.55, 0.35, 0.75])
    args['obstacle_r']      = 0.01
    args['alpha']           = np.array([1,3])

    args['pos_task_mode'] = 'track'
    args['ori_task_mode'] = 'track'

    args['T'] = 10

    args['pos_task_weight'] = 1
    args['pos_task_kp_track'] = 600
    args['pos_task_kd_track'] = 60
    args['pos_task_kd_damp'] = 20

    args['ori_task_weight'] = 1
    args['ori_task_kp_track'] = 10
    args['ori_task_kd_track'] = 1
    args['ori_task_kd_damp'] = 20


    main(args)