
import numpy as np
# import qpSWIFT as qp
from env.ur5_env import UR5Env
from qpsolvers import solve_qp
from .cbf import CBF

class InconsistentTaskSpaceController:


    def __init__(self, env : UR5Env, obstacle : np.ndarray = None, alpha : np.ndarray = None, obstacle_r = None, cbf=False):
        
        # mujoco parameters
        self.env = env

        self.cbf = cbf
        if self.cbf:
            self.cbf_filter = CBF(
                obstacle,
                alpha,
                obstacle_r
            )

    def get_eq_constraint(self,):
    
        '''
        H : mass matrix 6x6
        tau : actuated torques 6x1
        '''
        A = np.zeros((self.env.model.nv, self.env.model.nv + self.env.model.nu))
        b = np.zeros(self.env.model.nu)

        A = np.concatenate((-self.env.M, np.identity(6)), axis=1)

        b = self.env.C

        return A,b
    
    def get_ineq_constraint(self, tau_max):
        c_tau = np.ones((self.env.model.nu * 2,)) * tau_max
        C_tau = np.concatenate(
            [np.identity(6), -np.identity(6)]
        )

        if self.cbf:
            C_cbf, c_cbf = self.cbf_filter.get_cbf_ineq_constraints_q(
                self.env.ee_pos,
                self.env.data.qvel,
                self.env.jacp,
                self.env.M_inv,
                self.env.C
            )

            _C  = np.concatenate([C_tau, C_cbf])
            c   = np.concatenate([c_tau, c_cbf])

            C   = np.concatenate((np.zeros_like(_C), _C), axis=1)

        else:
            C = np.concatenate((np.zeros_like(C_tau), C_tau), axis=1)
            c = c_tau

        return C,c
    
    
    def get_action(self, g, H):

        g_qp = np.zeros((self.env.model.nv + self.env.model.nu,))
        H_qp = np.zeros((self.env.model.nv + self.env.model.nu, self.env.model.nv + self.env.model.nu))

        g_qp[:self.env.model.nv] = g
        H_qp[:self.env.model.nv,:self.env.model.nv] = H

        H_qp += np.identity(H_qp.shape[0]) * 1e-4

        A,b = self.get_eq_constraint()
        C,c = self.get_ineq_constraint(150)

        solution = solve_qp(P=H_qp, q=g_qp, A=A, b=b, G=C, h=c, solver="cvxopt", verbose=True)

        q_ddot, tau = solution[:self.env.model.nv], solution[self.env.model.nv: self.env.model.nv + self.env.model.nu]

        return q_ddot, tau

       