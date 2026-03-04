import numpy as np

class CLF:

    def __init__(self, P : np.ndarray, D : np.ndarray, alpha : np.float64):
        
        self.P      = P
        self.D      = D
        self.alpha  = alpha

    def V(self, x : np.ndarray, x_d : np.ndarray, xdot : np.ndarray, xdot_d : np.ndarray):

        e       = x_d - x
        edot    = xdot_d - xdot

        return (e.T @ self.P @ e + edot.T @ self.D @ edot) * 0.5

    def get_clf_ineq_constraints(self, x : np.ndarray, 
             x_d : np.ndarray, 
             xdot : np.ndarray, 
             xdot_d : np.ndarray,
             xddot_d : np.ndarray,
             J : np.ndarray,
             M_inv : np.ndarray,
             C : np.ndarray):
        
        e       = x_d - x
        edot    = xdot_d - xdot

        C_clf   = - edot.T @ self.D @ J @ M_inv

        c_clf   =   - self.alpha * self.V(x, x_d, xdot, xdot_d) \
                    - e.T @ self.P @ edot \
                    - edot.T @ self.D @ xddot_d \
                    + edot.T @ self.D @ J @ M_inv @ C
        
        return C_clf[np.newaxis, :], np.array([c_clf])