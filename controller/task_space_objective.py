import numpy as np
from utils.utils import *
from env.ur5_env import UR5Env
from utils.utils import skew_symmetric

class EEPositionTask():
    def __init__(self,env : UR5Env,w=50,Kp=800,Kd=20,Kd_contact=20,foot='left'):
        self.env = env
        self.Kp  = Kp*np.diag([1,1,1])
        self.Kd  = Kd*np.diag([1,1,1])
        self.Kd_contact = Kd_contact*np.diag([1,1,1])
        self.foot = foot
        
    def get_cmd_dynamics(self,ee_pos_ref=np.zeros(3,),ee_vel_ref=np.zeros(3,),ee_acc_ref=np.zeros(3,),mode='track'): # modes are track and damp

        if mode == 'track':
            cmd_task_dyn = ee_acc_ref + self.Kp @ (ee_pos_ref - self.env.ee_pos) + self.Kd @(ee_vel_ref - self.env.ee_vel)
        if mode == 'damp':
            cmd_task_dyn = self.Kd_contact@(ee_vel_ref - self.env.ee_vel)

        return cmd_task_dyn
            
class EEOrientationTask():
    def __init__(self,env : UR5Env,w=50,Kp=800,Kd=20,Kd_contact=20,foot='left'):
        self.env = env
        self.Kp  = Kp * np.diag([1,1,1])
        self.Kd  = Kd * np.diag([1,1,1])
        self.Kd_contact = Kd_contact * np.diag([1,1,1])
        self.foot = foot
    
    def get_cmd_dynamics(self,ee_q_ref=np.zeros(3,),ee_w_ref=np.zeros(3,),ee_wdot_ref=np.zeros(3,),mode='track'):
        
        foot_quaternion,foot_angular_velocity = self.env.ee_q, self.env.ee_w
        foot_ori = foot_quaternion
        foot_ori_ref = ee_q_ref

        e = self.get_quat_error(foot_ori, foot_ori_ref)
        
        if mode == 'track':
            cmd_task_dyn = ee_wdot_ref + self.Kp @ (e) + self.Kd @ (ee_w_ref - foot_angular_velocity)
        if mode == 'damp':
            cmd_task_dyn = self.Kd_contact @ (ee_w_ref - foot_angular_velocity)

        return cmd_task_dyn

    def get_euler_error(self, q, q_d):
        return quat2euler(q_d) - quat2euler(q)

    def get_quat_error(self, q, q_d):
        a = np.array(q_d[1:4])
        b = np.array(q[1:4])
        q_d_x = skew_symmetric(a)
        e = q[0]*a - q_d[0]*b - q_d_x @ b
        return e