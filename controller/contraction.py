import numpy as np

class Contraction:

    def __init__(self, 
                 Kp_pos : np.float64, 
                 Kd_pos : np.float64,
                 Kp_ori : np.float64, 
                 Kd_ori : np.float64
                 ):
        
        self.Kp = np.diag([Kp_pos, Kp_pos, Kp_pos, Kp_ori, Kp_ori, Kp_ori])
        self.Kd = np.diag([Kd_pos, Kd_pos, Kd_pos, Kd_ori, Kd_ori, Kd_ori])

    def _error_dynamics(self):

        # e_dot  = 0 I
        # e_ddot = -Kp(x_d - x) - Kd(x_dot_d - x_dot)

        A = np.block(
            [
                [np.zeros((6,6)),   np.identity(6)],
                [-self.Kp,          -self.Kd]
            ]
        )
        
        # A_sym = (A.T + A)/2

        eigs = np.linalg.eigvals(A)

        eig   = np.max(eigs)

        return A, eig
    
    def get_upper_bound(self, t, x, x_d, x_dot, x_dot_d):

        z = np.concatenate([x - x_d, x_dot - x_dot_d])

        z_norm = np.linalg.norm(z)

        _, eig = self._error_dynamics()

        exp = np.exp(eig * t)

        return z_norm, exp
    

    

