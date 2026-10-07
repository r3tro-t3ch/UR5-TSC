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
# Place the edge finder's block, the part below the camera, flat on the beam's
# end face: centered across the face's width, the top edge of its flat face
# offset_top below the end face's top edge, the tool axis pointing down along
# the face and the camera looking straight into it. The faces come from the
# newest measure_beam.py run (the touched end, top and side faces), so the beam
# must not have moved since. The arm moves in square to the face, stops at the
# first contact (the robot's own contact detection) and stays there.

import numpy as np
from pathlib import Path
from env.clearance import load_arm
from env.ur_robot import URRobot
from measure_beam import load, fit_faces, check_reach, tilt_plates
from utils.utils import pose_to_matrix, matrix_to_pose, transform_points

# the block below the camera (edge_finder.stl): its flat outer face looks along the flange's +y, as the camera, at
# y = 98.9 mm, 120 mm wide (x +-60 mm) and from 59.1 to 160.6 mm along the tool axis (a chamfer above it); the
# camera's glass is 3.6 mm further back. The middle of the flat face's top edge, in the flange frame (m)
BLOCK_TOP = np.array([0.0, 0.0989, 0.0591])


def run_faces(path : Path):
    # the touched faces of a measure_beam.py run (point, outward normal), and the beam frame from its camera;
    # None when its end or top face is missing or not every press on it seated
    _, T_beam, _, presses = load(path)
    planes  = fit_faces(presses)
    usable  = all(face in planes and planes[face]['flat'] for face in ('front', 'top'))
    return (planes, T_beam) if usable else None


def touched_faces(args):
    # the faces of the measure_beam.py run in beam_file, or of the newest results/beam_*.npz; when that one is not
    # usable the newest one that is gets named, to be set as beam_file if the beam has not moved since
    runs    = [Path(args['beam_file'])] if args['beam_file'] else sorted(Path(args['out_dir']).glob('beam_*.npz'))[::-1]
    faces   = run_faces(runs[0])
    if faces is None:
        usable = next((str(path) for path in runs[1:] if run_faces(path) is not None), None)
        raise SystemExit(f"{runs[0].name}: the end or top face was not touched with every press seated, run "
                         f"measure_beam.py again" + (f" (or set beam_file to {usable})" if usable else ""))
    return runs[0], *faces


def place_pose(planes : dict, T_beam : np.ndarray, args):
    # flange pose with the block's flat face on the end face: block and camera looking into it (flange y), the tool
    # axis down along it (flange z, from the top face), the middle of the block's top edge offset_top below the
    # middle of the end face's top edge. That middle is where the end face, the top face and the plane across
    # the width between the two side faces meet (through the camera's face center when a side was not touched)
    n_end, p_end = planes['front']['normal'], planes['front']['point']
    n_top, p_top = planes['top']['normal'], planes['top']['point']
    y       = -n_end
    z       = -(n_top - (n_top @ n_end) * n_end)
    z      /= np.linalg.norm(z)
    x       = np.cross(y, z)
    sides   = 'left' in planes and 'right' in planes
    middle  = (planes['left']['point'] + planes['right']['point']) / 2 if sides else T_beam[:3, 3]
    edge    = np.linalg.solve(np.vstack((n_end, n_top, x)), [n_end @ p_end, n_top @ p_top, x @ middle])
    T       = np.eye(4)
    T[:3, :3] = np.column_stack((x, y, z))
    T[:3, 3]  = edge + args['offset_top'] * z - T[:3, :3] @ BLOCK_TOP
    return T, edge, sides


def backed(T : np.ndarray, distance : float, normal : np.ndarray):
    # flange pose T moved distance (m) out along the end face's outward normal (< 0: into the face)
    T           = T.copy()
    T[:3, 3]   += distance * normal
    return T


def main(args):
    # the faces measured by touch, and where the block goes on the end face
    tilt_plates(args['plate_tilts'])        # the tool as measure_beam.py knows it, for its ball centers
    path, planes, T_beam = touched_faces(args)
    T_place, edge, sides = place_pose(planes, T_beam, args)
    n       = planes['front']['normal']
    print(f"faces from {path.name}: top edge middle {np.round(edge * 1000)} mm "
          f"({'between the touched sides' if sides else 'camera face center, a side was not touched'}), "
          f"block top edge {args['offset_top'] * 1000:.0f} mm below it")

    # in square to the face: a joint move to retract in front of it, a straight move to standoff, then the search
    # to search past it, all checked first (reach, joint limits, IK branch, singularities, the table)
    retract = backed(T_place, args['retract'], n)
    start   = backed(T_place, args['standoff'], n)
    limit   = backed(T_place, -args['search'], n)
    robot   = URRobot(args)
    robot.set_tcp([0, 0, 0, 0, 0, 0])       # every pose below is the flange's
    arm     = load_arm(args)
    try:
        q = robot.get_q()
        try:
            check_reach(robot, arm, [retract, start, limit], q, args, True)
        except RuntimeError as error:
            raise SystemExit(f"not reachable as planned, {error}")
        if not args['sim']:
            input(f"the block goes onto the end face measured in {path.name} (the beam must not have moved): "
                  "press Enter to move (Ctrl+C to stop) ")
        robot.move_j(robot.ik(matrix_to_pose(retract), q), args['move_speed'], args['move_acc'])
        robot.move_l(matrix_to_pose(start), args['lin_speed'], args['lin_acc'])

        # URSim feels no contact, but its beam is exactly where the simulated measurement put it: go to the pose
        if args['sim']:
            robot.move_l(matrix_to_pose(T_place), args['touch_speed'], args['touch_acc'])
            touched = True
        else:
            touched = robot.move_until_contact(matrix_to_pose(limit), args['touch_speed'], args['touch_acc'])

        # where the block's face is against the measured end face (+ = in front of it)
        T   = pose_to_matrix(robot.get_tcp_pose())
        gap = (transform_points(T, BLOCK_TOP[None])[0] - planes['front']['point']) @ n
        if touched:
            print(f"touched with the block's face {gap * 1000:+.1f} mm from the measured end face, staying there")
        else:
            print(f"no contact within {args['search'] * 1000:.0f} mm past the measured end face, backing off")
            robot.move_l(matrix_to_pose(start), args['lin_speed'], args['lin_acc'])
    except RuntimeError as error:
        print(f"{error}, stopping here: check the arm and the pendant")
    finally:
        robot.close()


if __name__ == "__main__":

    args = {}

    # real arm, or URSim (docker, ports on localhost); no camera is used
    args['sim']             = False
    args['robot_ip']        = "127.0.0.1" if args['sim'] else "192.168.1.100"

    # the measure_beam.py run whose touched faces say where the end face is: None = the newest results/beam_*.npz,
    # or a path. plate_tilts the same as in measure_beam.py (its tool model, for the ball centers)
    args['beam_file']       = None
    args['out_dir']         = 'results'
    args['plate_tilts']     = {'front': 0.0, 'top': 0.0}

    # where the block goes: the top edge of its flat face this far below the end face's top edge (m)
    args['offset_top']      = 0.010

    # approach along the end face's normal (m): joint move to retract in front of it, straight move to standoff,
    # then the contact search at touch_speed up to search past it
    args['retract']         = 0.15
    args['standoff']        = 0.04
    args['search']          = 0.02

    # speeds: touching (m/s, m/s^2), straight line moves, joint moves (rad/s, rad/s^2), as in measure_beam.py
    args['touch_speed']     = 0.01
    args['touch_acc']       = 0.2
    args['lin_speed']       = 0.1
    args['lin_acc']         = 0.3
    args['move_speed']      = 0.5
    args['move_acc']        = 0.5

    # checks on every step of the way, as in measure_beam.py: singularity margins (see visual_servo.py), the
    # table (top at table_z in UR's base frame, nothing closer than table_margin, shapes from xml_file), joint
    # turns from pose to pose, steps of step_len (m) and step_angle (rad)
    args['xml_file']        = 'ur10e.xml'
    args['run_margins']     = {'shoulder': 0.05, 'elbow': 0.10, 'wrist': 0.10}
    args['table_z']         = 0.0
    args['table_margin']    = 0.05
    args['max_joint_turn']  = np.radians(135)
    args['step_len']        = 0.01
    args['step_angle']      = np.radians(2)

    main(args)
