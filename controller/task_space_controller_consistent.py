
import numpy as np
# import qpSWIFT as qp
from env.ur5_env import UR5Env
from qpsolvers import solve_qp

class ConsistentTaskSpaceController:


    def __init__(self, env : UR5Env, obstacle : np.ndarray = None, alpha : np.ndarray = None, obstacle_r = None, cbf=False):
        
        # mujoco parameters
        self.env = env

        self.cbf = cbf
        if self.cbf:
            self.obstacle   = obstacle
            self.alpha      = alpha
            self.obstacle_r = obstacle_r
    
    def get_ineq_constraint(self, tau_max):

        J = np.concatenate([self.env.jacp, self.env.jacr])

        C = np.concatenate([J.T, -J.T])

        c = np.ones((self.env.model.nu * 2,)) * tau_max

        return C,c
    
    def h(self, x):
        # h(x) = (x - x_o)^2 - rq
        return (x - self.obstacle).T @ (x - self.obstacle) - self.obstacle_r**2

    def h_x(self, x):
        return 2*(x - self.obstacle)

    def h_dot(self, x):
        # h2(x) = Lfh(x) + alpha_1(h) 
        v = self.env.ee_vel
        return self.h_x(x).T @ v + self.alpha[0] * self.h(x)

    def get_cbf_ineq_constraints(self, x):
        # Lf^2 h(x) + LgLf h(x) >= -alpha_2(h2(x))
        v = self.env.ee_vel

        C_cbf = - self.h_x(x).T @ self.env.Lambda_inv[:3,:]
        c_cbf = - self.h_x(x).T @ (self.env.Lambda_inv @ self.env.mu)[:3] \
                + 2 * v.T @ v \
                - self.alpha[0] * self.h_x(x).T @ v \
                - self.alpha[1] * self.h_dot(x)

        return C_cbf[np.newaxis, :], np.array([c_cbf])
    
    def get_action(self, g, H):

        if self.cbf:
            C_tau, c_tau = self.get_ineq_constraint(150)
            C_cbf, c_cbf = self.get_cbf_ineq_constraints(self.env.ee_pos)

            C = np.concatenate([C_tau, C_cbf])
            c = np.concatenate([c_tau, c_cbf])

            print("h, hdot : ", self.h(self.env.ee_pos), self.h_dot(self.env.ee_pos))

        else:
            C, c = self.get_ineq_constraint(150)

        H += np.identity(H.shape[0]) + 1e-5

        J = np.concatenate([self.env.jacp, self.env.jacr])

        solution = solve_qp(P=H, q=g, G=C, h=c, solver="cvxopt", verbose=False)

        print("CBF ineq : ", )

        tau = J.T @ solution

        return tau

       