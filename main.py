# SPDX-License-Identifier: MIT
# MIT License
#
# Copyright (c) 2026 Vishnu Joshi
# Affiliation: CoRIS, Oregon State University
# Email: joshivis@oregonstate.edu
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Overview:
# Run the MuJoCo task-space tracking experiment with the selected torque QP
# controller and optional obstacle CBF. Log and plot end-effector position and
# orientation against the reference trajectory.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def main(args):
    from env.ur_env import UR10eEnv
    from controller.arm_controller import ArmController
    
    env = UR10eEnv(args)
    controller = ArmController(env, args)
    torq = np.zeros(env.n_joints)

    
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
    args['is_render']   = True
    args['control_mode'] = 'torque'  # These controllers output joint torques.
    args['xml_file']    = 'ur10e.xml'
    args['cam_azi']     = 90
    args['cam_ele']     = -20
    args['cam_dist']    =  5

    args['des_pos']     = np.array([0.8,0.8,0.8])
    args['des_ori_q']   = np.array([1, 0.0, 0.0, 0.0])

    # cbf
    args['cbf']             = True
    args['obstacle_pos']    = np.array([0.75, 0.5, 0.75])
    args['obstacle_r']      = 0.1
    args['alpha']           = np.array([50,100])

    args['position_task_mode']      = 'track'
    args['orientation_task_mode']   = 'track'

    args['T'] = 10

    args['position_task_weight']    = 1
    args['position_task_kp_track']  = 600
    args['position_task_kd_track']  = 60
    args['position_task_kd_damp']   = 20

    args['orientation_task_weight']     = 2
    args['orientation_task_kp_track']   = 600
    args['orientation_task_kd_track']   = 60
    args['orientation_task_kd_damp']    = 20

    args['use_pinnochio_dynamics']      = True

    # args['controller_type']             = 'inconsistent'
    args['controller_type']             = 'consistent'

    main(args)