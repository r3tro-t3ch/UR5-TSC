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
# Evaluate a quadratic task-space tracking-error Lyapunov candidate and
# assemble the associated torque inequality. Provide the algebra used by the
# experimental CLF-constrained controller.

import numpy as np

class CLF:

    def __init__(self, P : np.ndarray, D : np.ndarray, alpha : np.float64):
        
        self.P      = P
        self.D      = D
        self.alpha  = alpha

    def V(self, x : np.ndarray, x_d : np.ndarray, xdot : np.ndarray, xdot_d : np.ndarray):

        e       = x_d - x
        edot    = xdot_d - xdot

        return (e.T @ self.P @ e + edot.T @ self.D @ edot) * 0.5

    def get_clf_ineq_constraints(self, x : np.ndarray, 
             x_d : np.ndarray, 
             xdot : np.ndarray, 
             xdot_d : np.ndarray,
             xddot_d : np.ndarray,
             J : np.ndarray,
             M_inv : np.ndarray,
             C : np.ndarray):
        
        e       = x_d - x
        edot    = xdot_d - xdot

        C_clf   = - edot.T @ self.D @ J @ M_inv

        c_clf   =   - self.alpha * self.V(x, x_d, xdot, xdot_d) \
                    - e.T @ self.P @ edot \
                    - edot.T @ self.D @ xddot_d \
                    + edot.T @ self.D @ J @ M_inv @ C
        
        return C_clf[np.newaxis, :], np.array([c_clf])