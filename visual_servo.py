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
# face, and drive the camera with speedL until it looks at the face center from
# a set distance, a little from above. Press q or Esc to quit.

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from env.clearance import load_arm, joint_path, problems, table_problem
from env.ur_robot import URRobot
from env.vision import Vision
from test_vision import get_points, detect_faces, measure_face, draw_faces
from utils.utils import pose_to_matrix, ur_branch


def pixel_normals(points : np.ndarray, use : np.ndarray, w : int, k : int):
    # surface normal at every pixel, pointing at the camera (-z): the points are first averaged over w x w pixels,
    # only where use is set (so depth noise averages out and nothing off the surface pulls them), then differenced
    # k pixels to either side; zero in the outermost k pixels
    weight      = use.astype(np.float64)
    smooth      = cv2.blur(points * weight[..., None], (w, w)) / np.maximum(cv2.blur(weight, (w, w)), 1e-9)[..., None]
    dx          = np.zeros_like(points)
    dy          = np.zeros_like(points)
    dx[:, k:-k] = smooth[:, 2 * k:] - smooth[:, :-2 * k]
    dy[k:-k]    = smooth[2 * k:] - smooth[:-2 * k]
    n           = np.cross(dy, dx)
    return n / np.maximum(np.linalg.norm(n, axis=2, keepdims=True), 1e-12)


def find_wood_face(color : np.ndarray, depth : np.ndarray, intrinsics, args, rng):
    # the beam's end face. Color only points at the wood: the largest few planes through wood-colored pixels
    hsv     = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    wood    = cv2.inRange(hsv, args['hsv_low'], args['hsv_high']) > 0
    points  = get_points(depth, intrinsics)
    faces   = detect_faces(depth * wood, points, args, rng)

    # the end face looks back at the camera; the table (also wood) and the beam's top and sides are seen at a
    # slant, their normals more than max_face_angle from the optical axis (normals point at the camera, -z)
    faces   = [face for face in faces if -face['normal'][2] > np.cos(args['max_face_angle'])]
    if not faces:
        return None
    face    = max(faces, key=lambda face: face['mask'].sum())

    # the whole face, shadows included: every pixel on its plane whatever its color, whose own surface faces the
    # same way (within normal_tol), in the patch that holds the wood pixels. The table where it meets the plane at
    # the face's foot is on the plane too, but level. Opening removes speckle
    valid   = (depth > args['depth_min']) & (depth < args['depth_max'])
    offset  = -face['normal'] @ face['points'].mean(axis=0)
    plane   = valid & (np.abs(points @ face['normal'] + offset) < args['plane_tol'])
    normals = pixel_normals(points, plane, args['smooth_px'], args['normal_px'])
    facing  = normals @ face['normal'] > np.cos(args['normal_tol'])
    on      = cv2.morphologyEx((plane & facing).astype(np.uint8), cv2.MORPH_OPEN, np.ones((args['open_px'],) * 2, np.uint8))
    labels  = cv2.connectedComponents(on)[1]
    count   = np.bincount(labels[face['mask']], minlength=labels.max() + 1)
    count[0] = 0                        # background
    mask    = labels == np.argmax(count)
    return {'mask': mask, 'points': points[mask], 'normal': face['normal'], **measure_face(points[mask])}


def viewing_pose(face, T_base_cam : np.ndarray, args):
    # face center and normal (pointing at the camera) in the base frame
    center  = T_base_cam[:3, :3] @ face['midpoint'] + T_base_cam[:3, 3]
    normal  = T_base_cam[:3, :3] @ face['normal']

    # stand back from the face center along its normal turned view_elevation up (looking down at the face keeps
    # the tool, which hangs below the camera, high above the table)
    up      = np.array([0, 0, 1.0]) - normal[2] * normal
    up     /= np.linalg.norm(up)
    back    = np.cos(args['view_elevation']) * normal + np.sin(args['view_elevation']) * up

    # camera z looks at the face center, x stays as close as possible to the current x (no needless roll)
    z   = -back
    x   = T_base_cam[:3, 0] - (T_base_cam[:3, 0] @ z) * z
    x  /= np.linalg.norm(x)
    T           = np.eye(4)
    T[:3, :3]   = np.column_stack((x, np.cross(z, x), z))
    T[:3, 3]    = center + args['view_distance'] * back
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


def main(args):
    # refuse a home pose next to a singularity (speedL would fail there) or the table before anything starts
    arm     = load_arm(args)
    branch  = ur_branch(args['home_q'])
    near    = problems(np.array(args['home_q']), branch, arm, args['start_margins'], args)
    if near:
        raise SystemExit(f"home_q is not safe ({', '.join(near)}), move the arm and update it")

    # camera first: the first real frame and the window take over a second, which would trip the robot's watchdog
    vision  = Vision(args)
    cv2.imshow("visual servo", vision.get_frames()[0])
    cv2.waitKey(1)
    rng     = np.random.default_rng(0)

    # robot: make the camera the TCP, so TCP poses and speedL commands are the camera's
    robot   = URRobot(args)
    robot.set_tcp(args['cam_in_flange'])

    # start from home; the joint move there (which may turn the wrist over) must keep the arm off the table too
    for q in joint_path(robot.get_q(), np.array(args['home_q']), np.radians(2)):
        if table_problem(q, arm, args):
            robot.close()
            vision.close()
            raise SystemExit(f"the move to home_q takes the {table_problem(q, arm, args)[0]}, move the arm up "
                             "(freedrive) first")
    robot.move_j(args['home_q'], args['move_speed'], args['move_acc'])
    robot.start_watchdog(args['watchdog_hz'])     # robot stops if this loop stalls, so it is enabled last

    try:
        while True:
            robot.kick_watchdog()

            # stop before the arm reaches a singularity, where speedL finds no IK solution, or the table
            near = problems(robot.get_q(), branch, arm, args['run_margins'], args)
            if near:
                print(f"stopped ({', '.join(near)})")
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
    # both the real arm and URSim moveJ here before servoing. The wrist is above the tool (the other wrist
    # solution of the same pose, [1.5823, -0.9454, -2.4521, -1.3096, 1.5729, 3.1231], hangs it 18 cm below):
    # measure_beam.py keeps this wrist, and with the tool level at the beam's sides only this one clears the table
    args['home_q']          = [1.5823, -1.4434, -2.1595, 2.0374, -1.5729, -0.0185]

    # singularity guard (see utils.singularity_margins): the arm does not start when home_q is closer than
    # start_margins and stops servoing below run_margins. shoulder in m, elbow and wrist as |sin| (0.17 ~ 10 deg)
    args['start_margins']   = {'shoulder': 0.08, 'elbow': 0.17, 'wrist': 0.17}
    args['run_margins']     = {'shoulder': 0.05, 'elbow': 0.10, 'wrist': 0.10}
    args['move_speed']      = 0.5       # rad/s
    args['move_acc']        = 0.5       # rad/s^2

    # table the robot stands on: its top in UR's base frame (m, 0 = the base sits on it), and how close any part
    # of the arm or tool may come to it (env/clearance.py, collision shapes of xml_file)
    args['table_z']         = 0.0
    args['table_margin']    = 0.05

    # camera color optical frame in the flange frame [x, y, z, rx, ry, rz] (m, rotation vector)
    # from edge_finder.stl: D435 on the top pad, RGB lens 4.2 mm behind the front glass, looking along the
    # flange's +y with image down along the tool axis. CAD only, hand-eye calibrate before turning dry_run off
    # args['cam_in_flange']   = [0.0325, 0.0911, 0.0290, 0.0, 2.2214, 2.2214]
    args['cam_in_flange']   = [0.0314, 0.0891, 0.0293, -0.0328, -2.1095, -2.3137]

    # camera
    args['width']           = 640
    args['height']          = 480
    args['fps']             = 30

    # simulated scene (MuJoCo), poses in UR's base frame [x, y, z, rx, ry, rz]; the beam where the real one lay
    # on 2026-10-06 (end face center from the calibrated camera, width and height from touch), 3 cm over the table
    args['xml_file']        = 'ur10e.xml'
    args['sim_fovy']        = 43.1      # vertical field of view (deg), from the real D435's fy at 640x480
    args['beam_pose']       = [-0.012, 1.562, 0.161, 0, 0, 0]          # box center, end face at (-0.012, 0.812, 0.161)
    args['beam_size']       = [0.224, 1.5, 0.263]                      # width, length, height (m)
    args['beam_rgba']       = [0.8, 0.5, 0.2, 1]
    args['table_rgba']      = [0.75, 0.52, 0.28, 1]     # the real table is wood too

    # MuJoCo viewer window following the simulated arm (sim only), camera aimed at the arm and the beam
    args['render']          = True
    args['cam_lookat']      = [0.1, 0.9, 0.5]   # MuJoCo world frame (m), UR's base is 0.4 m above its origin
    args['cam_azi']         = 150               # deg
    args['cam_ele']         = -25               # deg
    args['cam_dist']        = 3.0               # m

    # wood color (OpenCV HSV: H 0-179), tuned on the beam end face in the lab
    args['hsv_low']         = np.array([8, 110, 40])
    args['hsv_high']        = np.array([25, 255, 255])

    # face detection (see test_vision.py): the largest of max_faces wood planes whose normal is within
    # max_face_angle of the optical axis (the table and the beam's other faces are seen at a slant), then every
    # pixel on its plane connected to it whose surface normal (points averaged over smooth_px, differenced
    # normal_px pixels apart) is within normal_tol of the face's, after an opening of open_px pixels
    args['depth_min']       = 0.3
    args['depth_max']       = 1.5
    args['max_faces']       = 3
    args['max_face_angle']  = np.radians(45)
    args['smooth_px']       = 9
    args['normal_px']       = 4
    args['normal_tol']      = np.radians(30)
    args['open_px']         = 5
    args['plane_tol']       = 0.01
    args['ransac_iters']    = 200
    args['ransac_points']   = 2000
    args['min_pixels']      = 3000

    # servoing
    args['view_distance']   = 0.4       # camera distance from the face center (m)
    args['view_elevation']  = np.radians(20)    # looking down at the face from this far above its normal
    args['gain']            = 0.2       # 1/s, the error shrinks by ~63 % every 5 s once below the speed caps
    args['max_speed']       = 0.02      # m/s
    args['max_ang_speed']   = 0.05      # rad/s (~3 deg/s)
    args['max_acc']         = 0.1       # m/s^2
    args['speed_time']      = 0.1       # s each speedL command is held, longer than one loop (17-30 Hz)
    args['pos_tol']         = 0.02      # m
    args['rot_tol']         = 0.05      # rad

    main(args)
