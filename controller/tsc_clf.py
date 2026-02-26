import numpy as np
from env.ur5_env import UR5Env
from qpsolvers import solve_qp
from .clf import TSCLF as CLF

class CLFTaskSpaceController:

    def __init__(self,  env : UR5Env, alpha : np.float64, P : np.ndarray, Q : np.ndarray):
        
        # mujoco parameters
        self.env        = env
        self.clf_filter = CLF(P,Q,alpha)

    def get_ineq_constraint(self, tau_max, x_d, xdot_d, delta_q, w_d, x_ddot_d):

        # Joint torque constraint
        J = np.concatenate([self.env.jacp, self.env.jacr])
        # C_tau = np.concatenate(
        #     [np.identity(6), -np.identity(6)]
        # )
        C_tau = np.concatenate(
            [J.T, -J.T]
        )
        c_tau = np.ones((self.env.model.nu * 2,)) * tau_max
        

        # CLF constraints
        C_clf, c_clf = self.clf_filter.get_clf_ineq_constraints(
            self.env.ee_pos,
            x_d,
            self.env.ee_vel,
            xdot_d,
            delta_q,
            self.env.ee_w,
            w_d,
            x_ddot_d,
            self.env.Lambda_inv,
            self.env.mu)

        C  = np.concatenate([C_tau, C_clf])
        c   = np.concatenate([c_tau, c_clf])

        return C,c
    
    def get_action(self, tau_max, x_d, xdot_d, delta_q, w_d, x_ddot_d, W):

        C, c = self.get_ineq_constraint(tau_max, x_d, xdot_d, delta_q, w_d, x_ddot_d)

        H = W + np.identity(W.shape[0]) * 1e-4
        # H = W
        g = np.zeros((self.env.model.nu,))

        tau = solve_qp(P=H, q=g, G=C, h=c, solver="cvxopt", verbose=False)

        return tau
    