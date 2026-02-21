import numpy as np
from utils.utils import *
from env.ur5_env import UR5Env
from utils.utils import skew_symmetric

class EEPositionTask():
    def __init__(self,env : UR5Env,w=1,Kp_track=800,Kd_track=20,Kd_damp=20):
        self.env = env
        self.Kp_track  = Kp_track*np.diag([1,1,1])
        self.Kd_track  = Kd_track*np.diag([1,1,1])
        self.Kd_damp = Kd_damp*np.diag([1,1,1])
        self.Q   = np.identity(3) * w
        
    def get_cost(self,ee_pos_ref=np.zeros(3,),ee_vel_ref=np.zeros(3,),ee_acc_ref=np.zeros(3,),mode='track'): # modes are track and damp

        if mode == 'track':
            cmd_task_dyn = ee_acc_ref + self.Kp_track @ (ee_pos_ref - self.env.ee_pos) + self.Kd_track @(ee_vel_ref - self.env.ee_vel)
        if mode == 'damp':
            cmd_task_dyn = self.Kd_damp@(ee_vel_ref - self.env.ee_vel)

        J = self.env.jacp

        H = J.T @ self.Q @ J
        g = (-J.T @ self.Q) @ cmd_task_dyn

        return H,g
            
class EEOrientationTask():
    def __init__(self,env : UR5Env,w=50,Kp_track=800,Kd_track=20,Kd_damp=20):
        self.env = env
        self.Kp  = Kp_track * np.diag([1,1,1])
        self.Kd  = Kd_track * np.diag([1,1,1])
        self.Kd_contact = Kd_damp * np.diag([1,1,1])
        self.Q   = np.identity(3) * w
    
    def get_cost(self,ee_q_ref=np.zeros(3,),ee_w_ref=np.zeros(3,),ee_wdot_ref=np.zeros(3,),mode='track'):
        
        foot_quaternion,foot_angular_velocity = self.env.ee_q, self.env.ee_w
        foot_ori = foot_quaternion
        foot_ori_ref = ee_q_ref

        e = self.get_quat_error(foot_ori, foot_ori_ref)
        
        if mode == 'track':
            cmd_task_dyn = ee_wdot_ref + self.Kp @ (e) + self.Kd @ (ee_w_ref - foot_angular_velocity)
        if mode == 'damp':
            cmd_task_dyn = self.Kd_contact @ (ee_w_ref - foot_angular_velocity)

        J = self.env.jacr

        H = J.T @ self.Q @ J
        g = (-J.T @ self.Q) @ cmd_task_dyn

        return H,g

    def get_euler_error(self, q, q_d):
        return quat2euler(q_d) - quat2euler(q)

    def get_quat_error(self, q, q_d):
        a = np.array(q_d[1:4])
        b = np.array(q[1:4])
        q_d_x = skew_symmetric(a)
        e = q[0]*a - q_d[0]*b - q_d_x @ b
        return e
    
class TaskConsistantEETask:

    def __init__(self,env : UR5Env,w=10,Kp_track=800,Kd_track=20,Kd_damp=20):
        self.env = env
        self.Kp_track   = Kp_track*np.diag([1,1,1,1,1,1])
        self.Kd_track   = Kd_track*np.diag([1,1,1,1,1,1])
        self.Kd_damp    = Kd_damp*np.diag([1,1,1,1,1,1])
        self.Q          = w * np.diag([1,1,1,2,2,2])

    def get_cost(self,ee_pos_ref=np.zeros(3,),ee_vel_ref=np.zeros(3,),ee_acc_ref=np.zeros(3,), ee_q_ref=np.zeros(3,),ee_w_ref=np.zeros(3,),ee_wdot_ref=np.zeros(3,),mode='track'):

        ee_dot_ref      = np.concatenate([ee_vel_ref, ee_w_ref])
        ee_ddot_ref     = np.concatenate([ee_acc_ref, ee_wdot_ref])

        delta_ee        = np.concatenate([ee_pos_ref - self.env.ee_pos, self.get_quat_error(self.env.ee_q, ee_q_ref)])
        delta_ee_dot    = ee_dot_ref - np.concatenate([self.env.ee_vel, self.env.ee_w])

        if mode == "track":
            x_ddot_ref = ee_ddot_ref + self.Kd_track @ (delta_ee_dot) + self.Kp_track @ delta_ee
        elif mode == "damp":
            x_ddot_ref = ee_ddot_ref + self.Kd_damp @ (delta_ee_dot)

        f_d = self.env.Lambda @ x_ddot_ref + self.env.mu

        g = -f_d.T @ self.Q
        H = self.Q

        return H, g, f_d

    def get_euler_error(self, q, q_d):
        return quat2euler(q_d) - quat2euler(q)

    def get_quat_error(self, q, q_d):
        a = np.array(q_d[1:4])
        b = np.array(q[1:4])
        q_d_x = skew_symmetric(a)
        e = q[0]*a - q_d[0]*b - q_d_x @ b
        return e


class TaskConsistantJointTask:

    def __init__(self, env : UR5Env, w=1, Kp_track=100, Kd_track=10, Kd_damp=10):

        self.env = env
        self.Kp_track   = np.identity(env.model.nu) * Kp_track
        self.Kd_track   = np.identity(env.model.nu) * Kd_track
        self.Kd_damp    = np.identity(env.model.nu) * Kd_damp
        
        self.Q          = np.identity(6) * w

    def get_cost(self, q_pos_ref, q_vel_ref, mode="track"):

        delta_pos_ref   = q_pos_ref - self.env.data.qpos
        delta_vel_ref   = q_vel_ref - self.env.data.qvel

        if mode == "track":
            q_ddot_ref  = self.Kp_track @ (delta_pos_ref) + self.Kd_track @ (delta_vel_ref)
        elif mode == "damp":
            q_ddot_ref  = self.Kd_damp @ (delta_vel_ref)

        J = np.concatenate([self.env.jacp, self.env.jacr])

        Gamma   = self.env.Lambda @ J

        f_d = Gamma @ q_ddot_ref + self.env.mu

        g = -f_d.T @ self.Q
        H = self.Q

        return H,g