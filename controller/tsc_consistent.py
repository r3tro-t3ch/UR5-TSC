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
# Map a nominal task-space wrench into joint torques and solve a minimum-
# deviation torque QP. Enforce actuator torque limits and optional obstacle
# CBF constraints, and require torque actuation mode.


import numpy as np
from env.ur_env import UR10eEnv
from qpsolvers import solve_qp
from .cbf import CBF

class ConsistentTaskSpaceController:

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

            C  = np.concatenate([C_tau, C_cbf])
            c   = np.concatenate([c_tau, c_cbf])

        else:
            C = C_tau
            c = c_tau

        return C,c
    
    def get_action(self, f_d):
        self.env.actuators.require_torque_mode()

        J = np.concatenate([self.env.jacp, self.env.jacr])
        
        # get joint torques
        tau = J.T @ f_d

        C, c = self.get_ineq_constraint()

        H = np.identity(self.env.n_joints)
        g = -tau.T

        tau_safe = solve_qp(P=H, q=g, G=C, h=c, solver="cvxopt", verbose=False)

        if tau_safe is None:
            raise RuntimeError("Torque QP failed; no command was applied")
        return tau_safe

       