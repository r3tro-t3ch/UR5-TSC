import numpy as np

class TSCLF:

    def __init__(self, P : np.ndarray, Q : np.ndarray, alpha : np.float64):
        self.P      = P
        self.Q      = Q
        self.alpha  = alpha

    def V(self, 
          x : np.ndarray, 
          x_d : np.ndarray, 
          xdot : np.ndarray, 
          xdot_d : np.ndarray,
          ori : np.ndarray,
          ori_d : np.ndarray,
          w : np.ndarray,
          w_d : np.ndarray):
        
        e        = np.concatenate([x_d - x,         ori_d - ori])
        e_dot    = np.concatenate([xdot_d - xdot,   w_d - w]) 

        return e.T @ self.P @ e + e_dot.T @ self.Q @ e_dot
    
    def V_x(self,
            x : np.ndarray, 
            x_d : np.ndarray, 
            xdot : np.ndarray, 
            xdot_d : np.ndarray,
            ori : np.ndarray,
            ori_d : np.ndarray,
            w : np.ndarray,
            w_d : np.ndarray):
    
        x     = np.concatenate([x, ori])
        x_d   = np.concatenate([x_d, ori_d])

        x_dot   = np.concatenate([xdot, w])
        x_dot_d = np.concatenate([xdot_d, w_d])

        return 2 * (x_d - x) @ self.P, 2 * (x_dot_d - x_dot).T @ self.Q
    
    def V_dot(self,
              x : np.ndarray, 
              x_d : np.ndarray,
              xdot : np.ndarray, 
              xdot_d : np.ndarray,
              ori : np.ndarray,
              ori_d : np.ndarray,
              w : np.ndarray,
              w_d : np.ndarray,
              x_ddot_d : np.ndarray,
              Lambda_inv : np.ndarray,
              mu : np.ndarray,
              ):
        
        V_x_1, V_x_2 = self.V_x(
            x=x, 
            x_d=x_d,
            xdot=xdot, 
            xdot_d=xdot_d,
            ori=ori,
            ori_d=ori_d,
            w=w,
            w_d=w_d
        )

        x_dot   = np.concatenate([xdot, w])
        x_dot_d = np.concatenate([xdot_d, w_d])

        LfV = V_x_1 @ (x_dot_d - x_dot) + V_x_2 @ (x_ddot_d + Lambda_inv @ mu)
        LgV =                             V_x_2 @ -Lambda_inv

        return LfV, LgV
    
    def get_clf_ineq_constraints(self,
                                 x : np.ndarray, 
                                 x_d : np.ndarray,
                                 xdot : np.ndarray, 
                                 xdot_d : np.ndarray,
                                 ori : np.ndarray,
                                 ori_d : np.ndarray,
                                 w : np.ndarray,
                                 w_d : np.ndarray,
                                 x_ddot_d : np.ndarray,
                                 Lambda_inv : np.ndarray,
                                 mu : np.ndarray,
                                 ):
        
        LfV, LgV = self.V_dot(
            x=x, 
            x_d=x_d,
            xdot=xdot, 
            xdot_d=xdot_d,
            ori=ori,
            ori_d=ori_d,
            w=w,
            w_d=w_d,
            x_ddot_d=x_ddot_d,
            Lambda_inv=Lambda_inv,
            mu=mu)
        
        V = self.V(
            x=x, 
            x_d=x_d,
            xdot=xdot, 
            xdot_d=xdot_d,
            ori=ori,
            ori_d=ori_d,
            w=w,
            w_d=w_d
        )

        C_clf = LgV

        c_clf = -self.alpha * V - LfV

        return C_clf[np.newaxis, :], np.array([c_clf])
    

class HOTSCLF:

    def __init__(self, P : np.ndarray, alpha_1 : np.float64, alpha_2 : np.float64):
        self.P          = P
        self.alpha_1    = alpha_1
        self.alpha_2    = alpha_2

    def V(self, x : np.ndarray, x_d : np.ndarray):

        e = x_d - x

        return e.T @ self.P @ e
    
    def V_x(self, x : np.ndarray, x_d : np.ndarray):

        e = x_d - x

        return 2 * e.T @ self.P
    
    def V1(self, x : np.ndarray, x_d : np.ndarray, xdot : np.ndarray, xdot_d : np.ndarray):

        edot = xdot_d - xdot

        LfV = self.V_x(x=x, x_d=x_d) @ edot 

        return LfV + self.alpha_1 + self.V(x=x, x_d=x_d)
    
    def get_clf_ineq_constraints(self, 
             x : np.ndarray, 
             x_d : np.ndarray, 
             xdot : np.ndarray, 
             xdot_d : np.ndarray,
             xddot_d : np.ndarray,
             Lambda_inv : np.ndarray,
             mu : np.ndarray):

        e       = x_d - x
        edot    = xdot_d - xdot

        Lf2V = 2 * edot.T @ self.P @ edot + 2 * self.alpha_1* e.T @ self.P @ edot \
            + 2 * e.T @ self.P @ (xddot_d + Lambda_inv @ mu)
        
        LgLfV = - 2 * e.T @ self.P @ Lambda_inv 

        C = LgLfV
        c = -self.alpha_2 * self.V1(x=x, x_d=x_d, xdot=xdot, xdot_d=xdot_d) - Lf2V

        return C[np.newaxis, :], np.array([c])