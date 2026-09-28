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
# Provide a manual scratch script for initializing a UR5e simulation and
# inspecting robot dynamics. Retain commented comparisons of MuJoCo and
# Pinocchio mass matrices, bias forces, and Jacobians; this is not an
# automated regression test.

import numpy as np
from env.ur_pinocchio_env import UR5EnvPinocchio
from env.ur_env import UR10eEnv
import mujoco


args = {}
args['is_render']   = True
args['xml_file']    = 'ur5e.xml'
args['cam_azi']     = 90
args['cam_ele']     = -20
args['cam_dist']    =  5

args['des_pos']     = np.array([0.6,0.6,0.6])
args['des_ori_q']   = np.array([1, 0.0, 0.0, 0.0])

# cbf
args['cbf']             = False

# pin_env     = UR5EnvPinocchio(args)
mj_env      = UR10eEnv(args)

q = np.ones(6) * 1.57
v = np.zeros(6)

# MuJoCo
mj_env.data.qpos[:] = q
mj_env.data.qvel[:] = v
# mujoco.mj_forward(mj_env.model, mj_env.data)
# mujoco.mj_crb(mj_env.model, mj_env.data)   # fill qM
mj_env.update_robot_states()               # recompute M, C, J, Lambda, mu

# # Pinocchio
# pin_env.set_state(q, v)

# M_pin = pin_env.M(q)

# M_mj = mj_env.M

# print("M : ", M_mj, M_pin)
# print("Difference:")
# print(M_pin - M_mj)

# print("Close?")
# print(np.allclose(M_pin, M_mj, atol=1e-5))

# print("C : ", mj_env.C, pin_env.C(q, v))
# print(np.allclose(mj_env.C, pin_env.C(q, v), atol=1e-5))

# J = np.concatenate([mj_env.jacp, mj_env.jacr])

# print(pin_env.J(q), J)
# print(np.allclose(pin_env.J(q), J, atol=1e-5))

