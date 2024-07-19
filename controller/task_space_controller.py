
import numpy as np
import qpSWIFT as qp
from env.ur5_env import UR5Env

class TaskSpaceController:


    def __init__(self, env : UR5Env):
        
        # mujoco parameters
        self.env = env

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
        c = np.ones((self.env.model.nu * 2,)) * tau_max
        C_tau = np.array([
            [-1,0,0,0,0,0],
            [1,0,0,0,0,0],
            [0,-1,0,0,0,0],
            [0,1,0,0,0,0],
            [0,0,-1,0,0,0],
            [0,0,1,0,0,0],
            [0,0,0,-1,0,0],
            [0,0,0,1,0,0],
            [0,0,0,0,-1,0],
            [0,0,0,0,1,0],
            [0,0,0,0,0,-1],
            [0,0,0,0,0,1]
        ],dtype=np.float32)

        C = np.concatenate((np.zeros_like(C_tau), C_tau), axis=1)

        return C,c
    
    
    def get_action(self, g, H):

        g_qp = np.zeros((self.env.model.nv + self.env.model.nu,))
        H_qp = np.zeros((self.env.model.nv + self.env.model.nu, self.env.model.nv + self.env.model.nu))

        g_qp[:self.env.model.nv] = g
        H_qp[:self.env.model.nv,:self.env.model.nv] = H

        A,b = self.get_eq_constraint()
        C,c = self.get_ineq_constraint(150)

        result = qp.run(g_qp, c, H_qp, C, A, b, opts={'MAXITER':30,'VERBOSE':0
                                                      ,'OUTPUT':1})
    
        solution = np.array(result['sol'])

        q_ddot, tau = solution[:self.env.model.nv], solution[self.env.model.nv: self.env.model.nv + self.env.model.nu]

        return q_ddot, tau

       