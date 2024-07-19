import numpy as np
import matplotlib.pyplot as plt

def main(args):
    from env.ur5_env import UR5Env
    from controller.arm_controller import ArmController
    
    env = UR5Env(args)
    controller = ArmController(env, args)
    torq = np.zeros((6,))
    
    while env.is_alive:
        env.step(torq)
        q_ddot, torq = controller.get_action()
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
    plt.legend(['ee_pos_x', 'ee_pos_x_ref'])

    plt.figure()
    plt.plot(times, ee_pos_y)
    plt.plot(times, ee_pos_y_ref)
    plt.legend(['ee_pos_y', 'ee_pos_y_ref'])

    plt.figure()
    plt.plot(times, ee_pos_z)
    plt.plot(times, ee_pos_z_ref)
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

    plt.show()



if __name__ == "__main__":

    args = {}
    args['is_render']  = True
    args['xml_file']   = 'ur5e.xml'
    args['cam_azi'] = 90
    args['cam_ele'] = -20
    args['cam_dist'] =  5

    args['des_pos'] = np.array([0.2,-0.1,0.8])
    args['des_ori_q'] = np.array([-1,1,0,0])

    args['pos_task_mode'] = 'track'
    args['ori_task_mode'] = 'track'

    args['T'] = 4

    args['pos_task_weight'] = 1
    args['pos_task_kp_track'] = 400
    args['pos_task_kd_track'] = 40
    args['pos_task_kd_damp'] = 20

    args['ori_task_weight'] = 1
    args['ori_task_kp_track'] = 400
    args['ori_task_kd_track'] = 40
    args['ori_task_kd_damp'] = 20


    main(args)