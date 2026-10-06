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
# Position-based visual servoing of a UR arm with a wrist-mounted camera, the
# same code for the real arm + RealSense and for URSim + MuJoCo (args['sim']).
# Find the wood by HSV color, fit a plane to the wood pixels to get the beam
# face, and drive the camera with speedL until it looks straight at the face
# center from a set distance. Press q or Esc to quit.

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from env.ur_robot import URRobot
from env.vision import Vision
from test_vision import get_points, detect_faces, draw_faces
from utils.utils import pose_to_matrix, singularity_margins


def find_wood_face(color : np.ndarray, depth : np.ndarray, intrinsics, args, rng):
    # wood pixels by color, then zero the depth of everything else so only wood can form a face
    hsv     = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    wood    = cv2.inRange(hsv, args['hsv_low'], args['hsv_high']) > 0
    faces   = detect_faces(depth * wood, get_points(depth, intrinsics), args, rng)
    return faces[0] if faces else None  # with max_faces = 1 this is the largest wood face


def viewing_pose(face, T_base_cam : np.ndarray, args):
    # face center and normal (pointing at the camera) in the base frame
    center  = T_base_cam[:3, :3] @ face['midpoint'] + T_base_cam[:3, 3]
    normal  = T_base_cam[:3, :3] @ face['normal']

    # camera z looks straight into the face, x stays as close as possible to the current x (no needless roll)
    z   = -normal
    x   = T_base_cam[:3, 0] - (T_base_cam[:3, 0] @ z) * z
    x  /= np.linalg.norm(x)

    # stand back from the face center along its normal
    T           = np.eye(4)
    T[:3, :3]   = np.column_stack((x, np.cross(z, x), z))
    T[:3, 3]    = center + args['view_distance'] * normal
    return T


def servo_velocity(T_target : np.ndarray, T_base_cam : np.ndarray, args):
    # pose error in the base frame: translation and rotation vector
    e_pos   = T_target[:3, 3] - T_base_cam[:3, 3]
    e_rot   = R.from_matrix(T_target[:3, :3] @ T_base_cam[:3, :3].T).as_rotvec()

    # proportional law, scaled down to the speed limits
    v   = args['gain'] * e_pos
    w   = args['gain'] * e_rot
    v  *= min(1.0, args['max_speed'] / max(np.linalg.norm(v), 1e-9))
    w  *= min(1.0, args['max_ang_speed'] / max(np.linalg.norm(w), 1e-9))

    done = np.linalg.norm(e_pos) < args['pos_tol'] and np.linalg.norm(e_rot) < args['rot_tol']
    return np.concatenate((v, w)), e_pos, e_rot, done


def near_singularity(q, limits):
    # the singularities closer than their limits, as readable text (empty when the arm is clear of all of them)
    margins = singularity_margins(q)
    return [f"{k} {margins[k]:.3f} < {limits[k]}" for k in limits if margins[k] < limits[k]]


def main(args):
    # refuse a home pose next to a singularity before anything starts: speedL would fail there
    near = near_singularity(args['home_q'], args['start_margins'])
    if near:
        raise SystemExit(f"home_q is too close to a singularity ({', '.join(near)}), move the arm and update it")

    # camera first: the first real frame and the window take over a second, which would trip the robot's watchdog
    vision  = Vision(args)
    cv2.imshow("visual servo", vision.get_frames()[0])
    cv2.waitKey(1)
    rng     = np.random.default_rng(0)

    # robot: make the camera the TCP, so TCP poses and speedL commands are the camera's
    robot   = URRobot(args)
    robot.set_tcp(args['cam_in_flange'])
    robot.move_j(args['home_q'], args['move_speed'], args['move_acc'])     # start from home
    robot.start_watchdog(args['watchdog_hz'])     # robot stops if this loop stalls, so it is enabled last

    try:
        while True:
            robot.kick_watchdog()

            # stop before the arm reaches a singularity, where speedL finds no IK solution
            near = near_singularity(robot.get_q(), args['run_margins'])
            if near:
                print(f"stopped, too close to a singularity ({', '.join(near)})")
                break

            color, depth    = vision.get_frames()
            face            = find_wood_face(color, depth, vision.intrinsics, args, rng)

            if face is None:
                robot.speed_l(np.zeros(6), args['max_acc'], args['speed_time'])     # lost the wood, hold still
                status, done = "no wood face", False
            else:
                T_base_cam              = pose_to_matrix(robot.get_tcp_pose())
                xd, e_pos, e_rot, done  = servo_velocity(viewing_pose(face, T_base_cam, args), T_base_cam, args)
                if not args['dry_run']:
                    robot.speed_l(xd, args['max_acc'], args['speed_time'])
                status = (f"err {np.linalg.norm(e_pos) * 100:.1f} cm {np.degrees(np.linalg.norm(e_rot)):.1f} deg  "
                          f"v {np.round(xd[:3] * 100, 1)} cm/s" + ("  DRY RUN" if args['dry_run'] else ""))

            # show the face and the servo state
            image = draw_faces(color, [face], vision.intrinsics) if face is not None else color.copy()
            cv2.putText(image, status, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow("visual servo", image)

            if done and not args['dry_run']:
                print("reached the viewing pose")
                break
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break
    finally:
        robot.close()
        vision.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":

    args = {}

    # real arm + RealSense, or URSim (docker, ports on localhost) + MuJoCo camera
    args['sim']             = False

    # robot
    args['robot_ip']        = "127.0.0.1" if args['sim'] else "192.168.1.100"
    args['dry_run']         = False      # compute and show the commands but do not move the arm
    args['watchdog_hz']     = 5         # minimum loop rate before the robot stops itself

    # home (rad): camera 25 cm in front of a pose read from the real arm on 2026-10-05, 18.5 cm clear of the
    # shoulder singularity; change it to start somewhere else (the guard below checks it).
    # both the real arm and URSim moveJ here before servoing
    args['home_q']          = [1.5823, -0.9454, -2.4521, -1.3096, 1.5729, 3.1231]

    # singularity guard (see utils.singularity_margins): the arm does not start when home_q is closer than
    # start_margins and stops servoing below run_margins. shoulder in m, elbow and wrist as |sin| (0.17 ~ 10 deg)
    args['start_margins']   = {'shoulder': 0.08, 'elbow': 0.17, 'wrist': 0.17}
    args['run_margins']     = {'shoulder': 0.05, 'elbow': 0.10, 'wrist': 0.10}
    args['move_speed']      = 0.5       # rad/s
    args['move_acc']        = 0.5       # rad/s^2

    # camera color optical frame in the flange frame [x, y, z, rx, ry, rz] (m, rotation vector)
    # from edge_finder.stl: D435 on the top pad, RGB lens 4.2 mm behind the front glass, looking along the
    # flange's +y with image down along the tool axis. CAD only, hand-eye calibrate before turning dry_run off
    args['cam_in_flange']   = [0.0325, 0.0911, 0.0290, 0.0, 2.2214, 2.2214]

    # camera
    args['width']           = 640
    args['height']          = 480
    args['fps']             = 30

    # simulated scene (MuJoCo), poses in UR's base frame [x, y, z, rx, ry, rz]
    args['xml_file']        = 'ur10e.xml'
    args['sim_fovy']        = 43.1      # vertical field of view (deg), from the real D435's fy at 640x480
    args['beam_pose']       = [-0.0741, 1.9744, 0.30, 0, 0, 0.2618]    # box center, end face at (0.12, 1.25, 0.30), 15 deg yaw
    args['beam_size']       = [0.19, 1.5, 0.35]                          # width, length, height (m), a guess
    args['beam_rgba']       = [0.8, 0.5, 0.2, 1]

    # MuJoCo viewer window following the simulated arm (sim only), camera aimed at the arm and the beam
    args['render']          = True
    args['cam_lookat']      = [0.1, 0.9, 0.5]   # MuJoCo world frame (m), UR's base is 0.4 m above its origin
    args['cam_azi']         = 150               # deg
    args['cam_ele']         = -25               # deg
    args['cam_dist']        = 3.0               # m

    # wood color (OpenCV HSV: H 0-179), tuned on the beam end face in the lab
    args['hsv_low']         = np.array([8, 110, 40])
    args['hsv_high']        = np.array([25, 255, 255])

    # face detection (see test_vision.py)
    args['depth_min']       = 0.3
    args['depth_max']       = 1.5
    args['max_faces']       = 1         # only the largest wood face
    args['plane_tol']       = 0.01
    args['ransac_iters']    = 200
    args['ransac_points']   = 2000
    args['min_pixels']      = 3000

    # servoing
    args['view_distance']   = 0.4       # camera distance from the face center (m)
    args['gain']            = 0.2       # 1/s, the error shrinks by ~63 % every 5 s once below the speed caps
    args['max_speed']       = 0.02      # m/s
    args['max_ang_speed']   = 0.05      # rad/s (~3 deg/s)
    args['max_acc']         = 0.1       # m/s^2
    args['speed_time']      = 0.1       # s each speedL command is held, longer than one loop (17-30 Hz)
    args['pos_tol']         = 0.02      # m
    args['rot_tol']         = 0.05      # rad

    main(args)
