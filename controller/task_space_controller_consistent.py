
import numpy as np
# import qpSWIFT as qp
from env.ur5_env import UR5Env
from qpsolvers import solve_qp

class ConsistentTaskSpaceController:


    def __init__(self, env : UR5Env):
        
        # mujoco parameters
        self.env = env
    
    def get_ineq_constraint(self, tau_max):

        J = np.concatenate([self.env.jacp, self.env.jacr])

        C = np.concatenate([J.T, -J.T])

        c = np.ones((self.env.model.nu * 2,)) * tau_max

        return C,c
    
    
    def get_action(self, g, H):

        C,c = self.get_ineq_constraint(150)
        J = np.concatenate([self.env.jacp, self.env.jacr])

        # f_des = -np.linalg.solve(H, g)
        # tau_des = J.T @ f_des
        # print("tau_des:", tau_des)

        solution = solve_qp(P=H, q=g, G=C, h=c, solver="cvxopt", verbose=True)


        print("Solution : ", solution.shape)

        tau = J.T @ solution

        return tau

       