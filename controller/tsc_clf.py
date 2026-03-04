import numpy as np
from env.ur5_env import UR5Env
from qpsolvers import solve_qp
from .clf import CLF

class CLFTaskSpaceController:

    def __init__(self,  env : UR5Env, alpha : np.float64, P : np.ndarray, D : np.ndarray):
        
        # mujoco parameters
        self.env        = env
        self.clf_filter = CLF(P, D, alpha)

    def get_ineq_constraint(self, tau_max, x_d, xdot_d, delta_q, w_d, x_ddot_d):

        # Joint torque constraint
        J = np.concatenate([self.env.jacp, self.env.jacr])
        
        # C_tau = np.concatenate(
        #     [J.T, -J.T]
        # )
        C_tau = np.concatenate(
            [np.identity(6), -np.identity(6)]
        )
        c_tau = np.ones((self.env.model.nu * 2,)) * tau_max
        
        _x      = np.concatenate([self.env.ee_pos, np.zeros((3,))])
        _x_d    = np.concatenate([x_d, delta_q])

        _xdot   = np.concatenate([self.env.ee_vel, self.env.ee_w])
        _xdot_d = np.concatenate([xdot_d, w_d])

        # CLF constraints
        C_clf, c_clf = self.clf_filter.get_clf_ineq_constraints(
            x=_x,
            x_d=_x_d,
            xdot=_xdot,
            xdot_d=_xdot_d,
            xddot_d=x_ddot_d,
            J=J,
            M_inv=self.env.M_inv,
            C=self.env.mu
        )

        C  = np.concatenate([C_tau, C_clf])
        c   = np.concatenate([c_tau, c_clf])

        # C = C_clf
        # c = c_clf

        return C,c
    
    def get_action(self, tau_max, x_d, xdot_d, delta_q, w_d, x_ddot_d, W, f_d):

        C, c = self.get_ineq_constraint(tau_max, x_d, xdot_d, delta_q, w_d, x_ddot_d)

        # Joint torque constraint
        J = np.concatenate([self.env.jacp, self.env.jacr])

        tau_nominal = J.T @ f_d

        H = W + np.identity(W.shape[0]) * 1e-4
        g = - tau_nominal.T @ W

        tau = solve_qp(P=H, q=g, G=C, h=c, solver="cvxopt", verbose=False)
    
        return tau