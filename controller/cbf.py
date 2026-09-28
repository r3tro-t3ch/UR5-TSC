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
# Evaluate a squared-distance barrier between an end-effector point and a
# stationary spherical obstacle. Assemble the current torque-space CBF
# inequality used by the task-space QPs; this implementation does not include
# the Jacobian-rate term.

import numpy as np

class CBF:

    def __init__(self, obstacle : np.ndarray = None, alpha : np.ndarray = None, obstacle_r = None):

        self.obstacle   = obstacle
        self.alpha      = alpha
        self.obstacle_r = obstacle_r

    def h(self, x):
        return (x - self.obstacle).T @ (x - self.obstacle) - self.obstacle_r**2
    
    def h_x_q(self, x : np.ndarray, J : np.ndarray):
        return 2*(x - self.obstacle) @ J

    def h_dot_q(self, x : np.ndarray, q_dot : np.ndarray, J : np.ndarray):
        return 2 * (x - self.obstacle).T @ J @ q_dot + self.alpha[0] * self.h(x)

    def get_cbf_ineq_constraints_q(self, x : np.ndarray, q_dot : np.ndarray, J : np.ndarray, M_inv : np.ndarray, C : np.ndarray):
        C_cbf = -2 * (x - self.obstacle).T @ J @ M_inv

        c_cbf =     self.alpha[1] * self.h_dot_q(x, q_dot, J) \
                +   2 * q_dot.T @ J.T @ J @ q_dot \
                +   self.alpha[0] * self.h_x_q(x, J) @ q_dot \
                -   self.h_x_q(x, J) @ M_inv @ C
        

        return C_cbf[np.newaxis, :], np.array([c_cbf])