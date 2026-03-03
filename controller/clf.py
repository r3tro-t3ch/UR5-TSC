import numpy as np

class TSCLF:

    def __init__(self, P : np.ndarray, D : np.ndarray, lam : np.float64):
        self.P          = P
        self.D          = D
        self.lam        = lam

    def V1(self, x : np.ndarray, x_d : np.ndarray, xdot : np.ndarray, xdot_d : np.ndarray):

        e       = x - x_d
        edot    = xdot - xdot_d

        return (e.T @ self.P @ e + edot.T @ self.D @ edot) * 0.5
    
    def V1_x(self, x : np.ndarray, x_d : np.ndarray, xdot : np.ndarray, xdot_d : np.ndarray):

        e       = x - x_d
        edot    = xdot - xdot_d

        return e.T @ self.P, edot.T @ self.D
    
    def get_clf_ineq_constraints(self, 
             x : np.ndarray, 
             x_d : np.ndarray, 
             xdot : np.ndarray, 
             xdot_d : np.ndarray,
             xddot_d : np.ndarray,
             Lambda_inv : np.ndarray,
             mu : np.ndarray):

        e       = x - x_d
        edot    = xdot - xdot_d
        v       = -Lambda_inv @ mu - xddot_d

        V1_x, V1_xdot = self.V1_x(x, x_d, xdot, xdot_d)
        
        LfV1    = V1_x @ edot + V1_xdot @ v

        V1_bar  = 2*self.lam*LfV1 + self.lam**2 * self.V1(x, x_d, xdot, xdot_d)

        LfLfV   = edot.T @ self.P @ edot + e.T @ self.P @ v + v.T @ self.D @ v
        LgLfV   = (e.T @ self.P + self.D @ v) @ Lambda_inv

        C = LgLfV
        c = - V1_bar - LfLfV

        return C[np.newaxis, :], np.array([c])