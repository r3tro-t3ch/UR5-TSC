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
# Filter nominal task-space tracking torques through a weighted QP with
# actuator limits and a CLF inequality. Require torque actuation mode and
# report solver failure instead of returning an absent command.

import numpy as np
from env.ur_env import UR10eEnv
from qpsolvers import solve_qp
from .clf import CLF

class CLFTaskSpaceController:

    def __init__(self,  env : UR10eEnv, alpha : np.float64, P : np.ndarray, D : np.ndarray):
        
        # mujoco parameters
        self.env        = env
        self.clf_filter = CLF(P, D, alpha)

    def get_ineq_constraint(self, tau_max, x_d, xdot_d, delta_q, w_d, x_ddot_d):

        J = np.concatenate([self.env.jacp, self.env.jacr])
        C_tau, c_tau = self.env.actuators.torque_constraints(tau_max)

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

        # C = C_tau
        # c = c_tau

        return C,c
    
    def get_action(self, tau_max, x_d, xdot_d, delta_q, w_d, x_ddot_d, W, f_d):

        self.env.actuators.require_torque_mode()
        C, c = self.get_ineq_constraint(tau_max, x_d, xdot_d, delta_q, w_d, x_ddot_d)

        # Joint torque constraint
        J = np.concatenate([self.env.jacp, self.env.jacr])

        tau_nominal = J.T @ f_d

        H = W + np.identity(W.shape[0]) * 1e-4
        g = - tau_nominal.T @ W

        tau = solve_qp(P=H, q=g, G=C, h=c, solver="cvxopt", verbose=False)
    
        if tau is None:
            raise RuntimeError("Torque QP failed; no command was applied")
        return tau