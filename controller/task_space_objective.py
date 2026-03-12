import numpy as np
from utils.utils import *
from env.armpi_env import ArmPiEnv
from utils.utils import skew_symmetric

class TaskConsistantEETask:

    def __init__(self,
                 env : ArmPiEnv,
                 Kp_track_pos=800,
                 Kd_track_pos=20,
                 Kd_damp_pos=20,
                 Kp_track_ori=800,
                 Kd_track_ori=20,
                 Kd_damp_ori=20):
        
        self.env = env
        self.Kp_track   = np.diag([Kp_track_pos,Kp_track_pos,Kp_track_pos,Kp_track_ori,Kp_track_ori,Kp_track_ori])
        self.Kd_track   = np.diag([Kd_track_pos,Kd_track_pos,Kd_track_pos,Kd_track_ori,Kd_track_ori,Kd_track_ori])
        self.Kd_damp    = np.diag([Kd_damp_pos,Kd_damp_pos,Kd_damp_pos,Kd_damp_ori,Kd_damp_ori,Kd_damp_ori])

    def get_cost(self,
                 ee_pos_ref=np.zeros(3,),
                 ee_vel_ref=np.zeros(3,),
                 ee_acc_ref=np.zeros(3,), 
                 ee_q_ref=np.zeros(3,),
                 ee_w_ref=np.zeros(3,),
                 ee_wdot_ref=np.zeros(3,),
                 pos_mode='track',
                 ori_mode='track'):

        ee_pos_dot_ref  = ee_vel_ref
        ee_ori_dot_ref  = ee_w_ref
        
        ee_pos_ddot_ref = ee_acc_ref
        ee_ori_ddot_ref = ee_wdot_ref

        delta_ee_pos        = ee_pos_ref - self.env.ee_pos if pos_mode == "track" else np.zeros((3,))
        delta_ee_ori        = self.get_quat_error(self.env.ee_q, ee_q_ref) if ori_mode == "track" else np.zeros((3,))

        delta_ee            = np.concatenate([delta_ee_pos, delta_ee_ori])

        delta_ee_pos_dot    = ee_pos_dot_ref - self.env.ee_vel
        delta_ee_ori_dot    = ee_ori_dot_ref - self.env.ee_w

        delta_ee_dot        = np.concatenate([delta_ee_pos_dot, delta_ee_ori_dot])

        Kd = np.zeros((6,6))
        Kd[:3,:3]   = self.Kd_track[:3,:3] if pos_mode == "track" else self.Kd_damp[:3,:3] 
        Kd[3:,3:]   = self.Kd_track[3:,3:] if ori_mode == "track" else self.Kd_damp[3:,3:] 

        ee_ddot_ref = np.concatenate([ee_pos_ddot_ref, ee_ori_ddot_ref])
        
        x_ddot_ref  = ee_ddot_ref + Kd @ delta_ee_dot + self.Kp_track @ delta_ee

        f_d = self.env.Lambda @ x_ddot_ref + self.env.mu

        return f_d

    def get_euler_error(self, q, q_d):
        return quat2euler(q_d) - quat2euler(q)

    def get_quat_error(self, q, q_d):
        a = np.array(q_d[1:4])
        b = np.array(q[1:4])
        q_d_x = skew_symmetric(a)
        e = q[0]*a - q_d[0]*b - q_d_x @ b
        return e

