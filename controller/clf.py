import numpy as np


class CLF:

    def __init__(self, P : np.ndarray, Q : np.ndarray, alpha : np.float64):
        self.P      = P
        self.Q      = Q
        self.alpha  = alpha

    def V(self, 
          x : np.ndarray, 
          x_d : np.ndarray, 
          xdot : np.ndarray, 
          xdot_d : np.ndarray,
          delta_q : np.ndarray,
          w : np.ndarray,
          w_d : np.ndarray):
        
        return (x_d - x).T @ self.P[:3,:3] @ (x_d - x) \
            + (xdot_d - xdot).T @ self.Q[:3, :3] @ (xdot_d - xdot) \
            + delta_q.T @ self.P[3:, 3:] @ delta_q \
            + (w_d - w).T @ self.Q[3:, 3:] @ (w_d - w)
    
    def V_x(self,
            x : np.ndarray, 
            x_d : np.ndarray, 
            xdot : np.ndarray, 
            xdot_d : np.ndarray,
            delta_q : np.ndarray,
            w : np.ndarray,
            w_d : np.ndarray,
            J : np.ndarray):
    
        delta_x     = np.concatenate([x_d-x, delta_q])
        _x_dot      = np.concatenate([xdot, w])
        _x_dot_d    = np.concatenate([xdot_d, w_d])

        return 2 * delta_x.T @ self.P @ J, 2 * (_x_dot_d - _x_dot).T @ self.Q @ J

    def V_dot(self,
              x : np.ndarray, 
              x_d : np.ndarray,
              xdot : np.ndarray, 
              xdot_d : np.ndarray,
              delta_q : np.ndarray,
              w : np.ndarray,
              w_d : np.ndarray,
              J : np.ndarray,
              M_inv : np.ndarray,
              C : np.ndarray,
              q_dot : np.ndarray):
        
        V_x_1, V_x_2 = self.V_x(
            x, 
            x_d,
            xdot, 
            xdot_d,
            delta_q,
            w,
            w_d,
            J
        )

        LfV = V_x_1 @ q_dot - V_x_2 @ (M_inv @ C)
        LgV =                 V_x_2 @ M_inv

        return LfV, LgV
    
    def get_clf_ineq_constraints(self,
                                 x : np.ndarray, 
                                 x_d : np.ndarray,
                                 xdot : np.ndarray, 
                                 xdot_d : np.ndarray,
                                 delta_q : np.ndarray,
                                 w : np.ndarray,
                                 w_d : np.ndarray,
                                 J : np.ndarray,
                                 M_inv : np.ndarray,
                                 C : np.ndarray,
                                 q_dot : np.ndarray):
        
        LfV, LgV = self.V_dot(x, 
            x_d,
            xdot, 
            xdot_d,
            delta_q,
            w,
            w_d,
            J,
            M_inv,
            C,
            q_dot)
        
        V = self.V(
            x, 
            x_d,
            xdot, 
            xdot_d,
            delta_q,
            w,
            w_d
        )

        C_clf = LgV

        c_clf = -self.alpha * V - LfV

        print("delta_q", delta_q)
        print("LgV", LgV)
        print("LfV", LfV)
        print("V", V)

        return C_clf[np.newaxis, :], np.array([c_clf])
