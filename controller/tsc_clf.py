import numpy as np
from env.ur5_env import UR5Env
from qpsolvers import solve_qp
from .clf import TSCLF as CLF

class CLFTaskSpaceController:

    def __init__(self,  env : UR5Env, alpha : np.float64, P : np.ndarray, Q : np.ndarray):
        
        # mujoco parameters
        self.env        = env
        self.clf_filter = CLF(P, Q, 3)

    def get_ineq_constraint(self, tau_max, x_d, xdot_d, ori_d, w_d, x_ddot_d):

        # Joint torque constraint
        J = np.concatenate([self.env.jacp, self.env.jacr])
        
        C_tau = np.concatenate(
            [J.T, -J.T]
        )
        c_tau = np.ones((self.env.model.nu * 2,)) * tau_max
        
        _x      = np.concatenate([self.env.ee_pos, self.env.ee_euler])
        _x_d    = np.concatenate([x_d, ori_d])

        _xdot   = np.concatenate([self.env.ee_vel, self.env.ee_w])
        _xdot_d = np.concatenate([xdot_d, w_d])

        # CLF constraints
        C_clf, c_clf = self.clf_filter.get_clf_ineq_constraints(
            x=_x,
            x_d=_x_d,
            xdot=_xdot,
            xdot_d=_xdot_d,
            xddot_d=x_ddot_d,
            Lambda_inv=self.env.Lambda_inv,
            mu=self.env.mu
        )

        C  = np.concatenate([C_tau, C_clf])
        c   = np.concatenate([c_tau, c_clf])

        # C = C_clf
        # c = c_clf

        return C,c
    
    def get_action(self, tau_max, x_d, xdot_d, ori_d, w_d, x_ddot_d, W):

        C, c = self.get_ineq_constraint(tau_max, x_d, xdot_d, ori_d, w_d, x_ddot_d)

        H = W + np.identity(W.shape[0]) * 1e-4
        g = np.zeros((self.env.model.nu,))

        f = solve_qp(P=H, q=g, G=C, h=c, solver="cvxopt", verbose=False)

        J = np.concatenate([self.env.jacp, self.env.jacr])

        return J.T @ f
    