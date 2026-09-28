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
# Construct an idealized linear second-order tracking-error dynamics matrix
# from position and orientation PD gains. Return the matrix and its maximum
# eigenvalue as diagnostics for the contraction experiment.

import numpy as np
from env.ur_pinocchio_env import UR5EnvPinocchio

class Contraction:

    def __init__(self, 
                 Kp_pos : np.float64, 
                 Kd_pos : np.float64,
                 Kp_ori : np.float64, 
                 Kd_ori : np.float64
                 ):
        
        self.Kp = np.diag([Kp_pos, Kp_pos, Kp_pos, Kp_ori, Kp_ori, Kp_ori])
        self.Kd = np.diag([Kd_pos, Kd_pos, Kd_pos, Kd_ori, Kd_ori, Kd_ori])

    def error_dynamics(self):

        # e_dot  = 0 I
        # e_ddot = -Kp(x_d - x) - Kd(x_dot_d - x_dot)

        A = np.block(
            [
                [np.zeros((6,6)),   np.identity(6)],
                [-self.Kp,          -self.Kd]
            ]
        )
        
        # A_sym = (A.T + A)/2

        eigs = np.linalg.eigvals(A)

        eig   = np.max(eigs)

        return A, eig
    

    

