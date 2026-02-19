
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
            self.obstacle_r = np.ones((6,)) * obstacle_r
    
    def get_ineq_constraint(self, tau_max):

        J = np.concatenate([self.env.jacp, self.env.jacr])

        C = np.concatenate([J.T, -J.T])

        c = np.ones((self.env.model.nu * 2,)) * tau_max

        return C,c
    
    def h(self, x):
        # h(x) = x - (x_o + r)
        return x - self.obstacle - self.obstacle_r

    def h2(self, x):
        # h2(x) = Lfh(x) + alpha_1(h) 
        return np.concatenate([self.env.ee_vel, self.env.ee_w]) + self.alpha[0] * self.h(x)

    def h_ddot(self,):
        # h_ddot = -Lambda_inv mu + Lambda_inv J^T
        J = np.concatenate([self.env.jacp, self.env.jacr])
        return -self.env.Lambda_inv @ self.env.mu + self.env.Lambda_inv @ J.T

    def get_cbf_ineq_constraints(self, x):
        # Lf^2 h(x) + LgLf h(x) >= -alpha_2(h2(x))
        C_cbf = self.h_ddot()
        c_cbf = - self.alpha[1] * self.h2(x)

        return C_cbf, c_cbf
    
    def get_action(self, g, H):


        if self.cbf:
            C_tau, c_tau = self.get_ineq_constraint(150)
            C_cbf, c_cbf = self.get_cbf_ineq_constraints(np.array([*self.env.ee_pos, *np.zeros((3,))]))

            # print(C_cbf)

            C = np.concatenate([C_tau, C_cbf])
            c = np.concatenate([c_tau, c_cbf])

        else:
            C, c = self.get_ineq_constraint(150)


        J = np.concatenate([self.env.jacp, self.env.jacr])

        solution = solve_qp(P=H, q=g, G=C, h=c, solver="cvxopt", verbose=True)

        tau = J.T @ solution

        return tau

       