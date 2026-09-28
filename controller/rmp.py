import numpy as np
from utils.utils import *
from env.ur_env import UR10eEnv

class RMP:

    def __init__(self, 
                 alpha      : np.float64,
                 beta       : np.float64,
                 Kp_ori     : np.float64,
                 Kd_ori     : np.float64,
                 alpha_obs  : np.float64,
                 beta_obs   : np.float64,
                 env        : UR10eEnv,
                 obstacles  : np.ndarray,
                 obstacle_r : np.float64,
                 c          : np.float64
                 ):
        
        # attractor gains
        self.alpha  = alpha
        self.beta   = beta
        self.Kp_ori = Kp_ori
        self.Kd_ori = Kd_ori

        self.c      = c

        # env
        self.env    = env
        
        # obstacle and radius
        self.obstacles  = obstacles
        self.r          = obstacle_r
        self.alpha_obs  = alpha_obs
        self.beta_obs   = beta_obs


    def position_tracking_policy(self, target_pos):
        f = self.alpha * self.s(target_pos - self.env.ee_pos) - self.beta * self.env.ee_vel
        A = np.identity(3)
        return f, A

    def s(self, v : np.ndarray):
        return v/self.h(np.linalg.norm(v))

    def h(self, z : np.float64):
        return z + self.c * np.log(1 + np.exp(-2*self.c*z)) 

    def orientation_tracking_policy(self, orientation_desired):
        f = self.Kp_ori * get_quat_error(self.env.ee_q, orientation_desired) - self.Kd_ori * self.env.ee_w
        A = np.identity(3)

        return f, A
    
    def collision_policy(self, obs_pos):

        v = self.env.ee_pos - obs_pos

        d = np.linalg.norm(v)
        
        v_hat = v/d

        f = self.alpha_obs * d * v_hat - self.beta_obs * d * (v_hat @ v_hat.T) @ self.env.ee_vel

        def w(d):
            return np.exp(-d)

        A = w(d) * self.s(f) @ self.s(f).T

        return f, A
    
    