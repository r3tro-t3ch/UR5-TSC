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
# orientation tracking calculations. Convert UR poses (position and rotation
# vector) to 4x4 transforms and back. UR forward kinematics (DH), distances
# to the arm's three singularities (shoulder, elbow, wrist) and which of the
# eight IK solutions (branches) a joint configuration is.

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

def pose_to_matrix(pose):
    # UR pose [x, y, z, rx, ry, rz] (rotation vector) -> 4x4 transform
    T           = np.eye(4)
    T[:3, :3]   = R.from_rotvec(pose[3:]).as_matrix()
    T[:3, 3]    = pose[:3]
    return T

def matrix_to_pose(T):
    # 4x4 transform -> UR pose [x, y, z, rx, ry, rz] (rotation vector)
    return np.concatenate((T[:3, 3], R.from_matrix(T[:3, :3]).as_rotvec()))

def transform_points(T, points):
    # Nx3 points through a 4x4 transform
    return points @ T[:3, :3].T + T[:3, 3]

# UR10e DH parameters (UR's published values): d (m), a (m), alpha (rad) per joint
UR10E_DH = {'d'     : [0.1807, 0, 0, 0.17415, 0.11985, 0.11655],
            'a'     : [0, -0.6127, -0.57155, 0, 0, 0],
            'alpha' : [np.pi / 2, 0, 0, np.pi / 2, -np.pi / 2, 0]}

def ur_forward_kinematics(q, dh=UR10E_DH):
    # joint angles (rad) -> 4x4 flange pose in UR's base frame (same as the robot's TCP pose with no TCP offset)
    T = np.eye(4)
    for qi, d, a, al in zip(q, dh['d'], dh['a'], dh['alpha']):
        ct, st, ca, sa = np.cos(qi), np.sin(qi), np.cos(al), np.sin(al)
        T = T @ np.array([[ct, -st * ca,  st * sa, a * ct],
                          [st,  ct * ca, -ct * sa, a * st],
                          [0,   sa,       ca,      d],
                          [0,   0,        0,       1]])
    return T

def singularity_margins(q, dh=UR10E_DH):
    # how far the arm is from each singularity, where IK (and so speedL) breaks down:
    # shoulder: wrist point (frame 5 origin) distance outside the cylinder of radius d4 around the base axis (m),
    #           inside it there is no IK solution at all
    # elbow:    |sin(q3)|, 0 with the arm stretched out or folded back
    # wrist:    |sin(q5)|, 0 with the wrist 1 and wrist 3 axes lined up
    T   = ur_forward_kinematics(q, dh)
    p05 = T[:3, 3] - dh['d'][5] * T[:3, 2]
    return {'shoulder'  : np.hypot(p05[0], p05[1]) - dh['d'][3],
            'elbow'     : abs(np.sin(q[2])),
            'wrist'     : abs(np.sin(q[4]))}

def ur_branch(q, dh=UR10E_DH):
    # which of the eight IK solutions q is, as signs (shoulder, elbow, wrist): the wrist point in front of or
    # behind the base along the arm's plane, and the signs of sin(q3) and sin(q5). Between two configurations of
    # the same branch the arm can move without crossing a singularity; across branches it swings far
    T   = ur_forward_kinematics(q, dh)
    p05 = T[:3, 3] - dh['d'][5] * T[:3, 2]
    return (int(np.sign(p05[0] * np.cos(q[0]) + p05[1] * np.sin(q[0]))), int(np.sign(np.sin(q[2]))),
            int(np.sign(np.sin(q[4]))))

def get_quat_error(q, q_d):
        a = np.array(q_d[1:4])
        b = np.array(q[1:4])
        q_d_x = skew_symmetric(a)
        e = q[0]*a - q_d[0]*b - q_d_x @ b
        return e
