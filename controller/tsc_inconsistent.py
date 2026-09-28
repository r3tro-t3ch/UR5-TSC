# SPDX-License-Identifier: MIT
# MIT License
#
# Copyright (c) 2026 Vishnu Joshi
# Affiliation: CoRIS, Oregon State University
# Email: joshivis@oregonstate.edu
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Overview:
# Solve a joint-acceleration and torque QP using task tracking costs and a
# robot dynamics equality. Enforce actuator torque bounds and optional
# obstacle CBF inequalities, and require torque actuation mode.


import numpy as np
# import qpSWIFT as qp
from env.ur_env import UR10eEnv
from qpsolvers import solve_qp
from .cbf import CBF

class InconsistentTaskSpaceController:


    def __init__(self, env : UR10eEnv, obstacle : np.ndarray = None, alpha : np.ndarray = None, obstacle_r = None, cbf=False):
        
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
        A = np.zeros((self.env.model.nv, self.env.model.nv + self.env.n_joints))
        b = np.zeros(self.env.n_joints)

        A = np.concatenate((-self.env.M, np.identity(self.env.n_joints)), axis=1)

        b = self.env.C

        return A,b
    
    def get_ineq_constraint(self, tau_max=None):
        C_tau, c_tau = self.env.actuators.torque_constraints(tau_max)

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
        self.env.actuators.require_torque_mode()

        g_qp = np.zeros((self.env.model.nv + self.env.n_joints,))
        H_qp = np.zeros((self.env.model.nv + self.env.n_joints, self.env.model.nv + self.env.n_joints))

        g_qp[:self.env.model.nv] = g
        H_qp[:self.env.model.nv,:self.env.model.nv] = H

        H_qp += np.identity(H_qp.shape[0]) * 1e-4

        A,b = self.get_eq_constraint()
        C,c = self.get_ineq_constraint()

        solution = solve_qp(P=H_qp, q=g_qp, A=A, b=b, G=C, h=c, solver="cvxopt", verbose=True)

        if solution is None:
            raise RuntimeError("Torque QP failed; no command was applied")

        q_ddot, tau = solution[:self.env.model.nv], solution[self.env.model.nv: self.env.model.nv + self.env.n_joints]

        return q_ddot, tau

       