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
# What keeps the arm from a joint configuration: no IK solution, a joint past
# its limit, another IK branch than where it started, a singularity, or any part
# of the arm or the edge finder near the table the robot stands on (from the
# collision shapes of the MuJoCo model: capsules and cylinders around the links,
# boxes and balls of the tool). Also the joint angles a joint move passes
# through, so the moves between poses can be checked as well as the poses.

import mujoco
import numpy as np
from pathlib import Path
from utils.utils import singularity_margins, ur_branch

# the parts that move above the table: everything from the upper arm on (the base and shoulder only turn in place)
MOVING = ['upper_arm_link', 'forearm_link', 'wrist_1_link', 'wrist_2_link', 'wrist_3_link', 'edge_finder']


def load_arm(args):
    # the arm model and its collision shapes on the moving links (MuJoCo's collision group 3)
    model   = mujoco.MjModel.from_xml_path(str(Path(__file__).resolve().parent / args['xml_file']))
    bodies  = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in MOVING]
    geoms   = [g for g in range(model.ngeom) if model.geom_group[g] == 3 and model.geom_bodyid[g] in bodies]
    return {'model': model, 'data': mujoco.MjData(model), 'geoms': geoms,
            'base': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'base')}


def lowest_point(arm : dict, q : np.ndarray):
    # height (m, UR base frame) of the lowest point of the moving arm and tool at joint angles q, and its link
    model, data = arm['model'], arm['data']
    data.qpos[:] = q
    mujoco.mj_kinematics(model, data)
    best = (np.inf, None)
    for g in arm['geoms']:
        z, R, size = data.geom_xpos[g][2], data.geom_xmat[g].reshape(3, 3), model.geom_size[g]
        a = abs(R[2, 2])            # how vertical the shape's own z axis (a capsule's or cylinder's axis) is
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE:
            bottom = z - size[0]
        elif model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE:
            bottom = z - size[1] * a - size[0]
        elif model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER:
            bottom = z - size[1] * a - size[0] * np.sqrt(1 - a * a)
        else:                       # box
            bottom = z - np.abs(R[2]) @ size
        if bottom < best[0]:
            best = (bottom, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]))
    # UR's base frame is the MuJoCo base body turned about z, so heights only shift by the base's height
    return best[0] - data.xpos[arm['base']][2], best[1]


def joint_path(q_from : np.ndarray, q_to : np.ndarray, step : float):
    # joint angles a joint move (moveJ, every joint linear in time) passes through, no joint turning more than
    # step (rad) between two of them, the end included
    n = max(1, int(np.ceil(np.abs(q_to - q_from).max() / step)))
    return [q_from + (q_to - q_from) * (i + 1) / n for i in range(n)]


def near_singularity(q, limits):
    # the singularities closer than their limits, as readable text (empty when the arm is clear of all of them)
    margins = singularity_margins(q)
    return [f"{k} {margins[k]:.3f} < {limits[k]}" for k in limits if margins[k] < limits[k]]


def problems(q, branch : tuple, arm : dict, margins : dict, args):
    # what keeps the arm from joint angles q, as readable text (empty when nothing): no IK solution (None), a joint
    # past +-360 deg (the controller's IK can return more), another IK branch than branch (shoulder, elbow, wrist;
    # the nearest IK solution can be one), a singularity closer than margins, or a part of the arm or tool less
    # than table_margin above the table top (table_z in UR's base frame)
    if q is None:
        return ['out of reach']
    if np.abs(q).max() > 2 * np.pi:
        return [f"joint {np.argmax(np.abs(q)) + 1} past its limit"]
    if ur_branch(q) != branch:
        return [f"IK branch {ur_branch(q)} instead of {branch} (shoulder, elbow, wrist)"]
    return near_singularity(q, margins) + table_problem(q, arm, args)


def table_problem(q, arm : dict, args):
    # a part of the arm or tool less than table_margin above the table top (table_z in UR's base frame), as text;
    # all that matters on the way of a joint move, where singularities and IK branches do not
    height, link = lowest_point(arm, q)
    gap          = height - args['table_z']
    return [f"{link} {gap * 1000:.0f} mm above the table"] if gap < args['table_margin'] else []
