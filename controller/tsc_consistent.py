
import numpy as np
from env.armpi_env import ArmPiEnv
from qpsolvers import solve_qp

class ConsistentTaskSpaceController:

    def __init__(self, env : ArmPiEnv):
        
        # mujoco parameters
        self.env = env

    def get_ineq_constraint(self, tau_max):
        C_tau = np.concatenate(
            [np.identity(5), -np.identity(5)]
        )
        c_tau = np.ones((self.env.model.nu * 2,)) * tau_max
    
        C = C_tau
        c = c_tau

        return C,c
    
    def get_action(self, f_d):

        J = np.concatenate([self.env.jacp, self.env.jacr])
        
        # get joint torques
        tau = J.T @ f_d

        C, c = self.get_ineq_constraint(5)

        H = np.identity(5)
        g = -tau.T

        tau_safe = solve_qp(P=H, q=g, G=C, h=c, solver="cvxopt", verbose=False)

        return tau_safe

       