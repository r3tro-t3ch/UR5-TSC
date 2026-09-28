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
# Provide quaternion-to-Euler conversion for scalar-first input quaternions
# and construction of three-dimensional skew-symmetric matrices used by
# orientation tracking calculations.

import numpy as np
from scipy.spatial.transform import Rotation as R

def quat2euler(quat):
    _quat = np.concatenate([quat[1:], quat[:1]])
    r = R.from_quat(_quat)
    euler = r.as_euler('xyz', degrees=False)
    return euler

def skew_symmetric(vector):
    mat = np.zeros((vector.shape[0], vector.shape[0]))

    mat[0,1] = -vector[2]
    mat[0,2] = vector[1]
    mat[1,0] = vector[2]
    mat[1,2] = -vector[0]
    mat[2,0] = -vector[1]
    mat[2,1] = vector[0]

    return mat
