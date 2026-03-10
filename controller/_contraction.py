import numpy as np
from env.ur5_pinocchio_env import UR5EnvPinocchio

class Contraction:

    def __init__(self, 
                 Kp_pos : np.float64, 
                 Kd_pos : np.float64,
                 Kp_ori : np.float64, 
                 Kd_ori : np.float64
                 ):
        
        self.Kp = np.diag([Kp_pos, Kp_pos, Kp_pos, Kp_ori, Kp_ori, Kp_ori])
        self.Kd = np.diag([Kd_pos, Kd_pos, Kd_pos, Kd_ori, Kd_ori, Kd_ori])

    def error_dynamics(self, x_d, x, x_dot_d, x_dot):

        # e_dot  = 0 I
        # e_ddot = -Kp(x_d - x) - Kd(x_dot_d - x_dot)

        A = np.block(
            [
                [np.zeros((6,6)),   np.identity(6)],
                [-self.Kp,          -self.Kd]
            ]
        )

        A_sym = (A.T + A) * 0.5

        eigs = np.linalg.eigvalsh(A_sym)

        lam_min = np.maximum(eigs[0], 1e-10)   # clip near-zero
        lam_max = eigs[-1]
        kappa = float(np.sqrt(lam_max / lam_min))
        
        eig   = np.max(eigs)

        return A, eig, kappa
    

    

