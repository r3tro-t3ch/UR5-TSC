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
# Hand-eye calibration of the wrist camera (eye in hand): where the camera's
# color optical frame sits in the flange frame (cam_in_flange). A ChArUco board
# stays still in the workspace. From a start pose where the camera sees it, the
# arm visits views tilted and rolled around the board's center; at each one the
# flange pose (from the robot) and the board corners (from the image) are kept.
# One least-squares fit of the camera mount and the board's pose then minimizes
# the corners' reprojection error over all views, starting from the CAD mount.
# Same code for the real arm + RealSense and for URSim + MuJoCo, where the board
# is rendered and the camera sits at a known mount that differs from the CAD.
# Set print_board to write the board as a PDF to print instead.

import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
from env.clearance import load_arm, joint_path, problems, table_problem
from env.ur_robot import URRobot
from env.vision import Vision
from utils.charuco import make_board, board_image, detect_board
from utils.utils import pose_to_matrix, matrix_to_pose, ur_branch


def save_board_pdf(board, args):
    # the board at its true size centered on one page (print at 100 % / actual size), with a 100 mm bar to check
    # the printer's scale and a line saying what the board is
    page    = np.array(args['paper_mm']) / 25.4                                             # in
    size    = np.array(args['board_squares']) * args['square_len'] / 0.0254                 # in
    fig     = plt.figure(figsize=page)
    ax      = fig.add_axes([*((page - size) / 2 / page), *(size / page)])
    ax.imshow(board_image(board, args, args['print_px_per_square'], 0.0), cmap='gray', vmin=0, vmax=255,
              interpolation='nearest', aspect='auto')
    ax.axis('off')

    # 100 mm bar and the label in the bottom margin
    bottom  = (page[1] - size[1]) / 2 / page[1]
    left    = (page[0] - size[0]) / 2 / page[0]
    fig.add_artist(plt.Line2D([left, left + 100 / 25.4 / page[0]], [bottom / 2] * 2, color='k', lw=1))
    fig.text(left, bottom / 2 + 0.01, "100 mm", fontsize=7)
    fig.text(left + 120 / 25.4 / page[0], bottom / 2, f"ChArUco {args['board_squares'][0]} x {args['board_squares'][1]}, "
             f"{args['board_dict']}, square {args['square_len'] * 1000:g} mm, marker {args['marker_len'] * 1000:g} mm. "
             "Print at 100 % (actual size)", fontsize=7, va='center')
    fig.savefig(args['print_board'])
    print(f"saved {args['print_board']}: print it at 100 %, glue it flat on something rigid, then measure "
          f"{args['board_squares'][0]} squares with calipers and set square_len (and marker_len in proportion)")


def camera_matrix(intrinsics):
    # pinhole camera matrix; the D435 color stream reports zero distortion, so none is modelled
    return np.array([[intrinsics.fx, 0, intrinsics.ppx], [0, intrinsics.fy, intrinsics.ppy], [0, 0, 1]])


def capture(vision, detector, board, args):
    # the board corners in a fresh frame once the arm is still (older frames queued up during the move are
    # dropped), shown in a window; None when too few corners are found
    for _ in range(args['flush_frames']):
        color, _ = vision.get_frames()
    found   = detect_board(detector, board, color, args['min_corners'])
    image   = color.copy()
    if found is not None:
        for u, v in found['uv']:
            cv2.circle(image, (int(round(u)), int(round(v))), 3, (0, 255, 0), -1)
    cv2.imshow("hand eye", image)
    cv2.waitKey(1)
    return None if found is None else dict(found, color=color)


def board_in_camera(view : dict, K : np.ndarray):
    # board pose in the camera frame from one view's corners (planar PnP)
    _, rvec, tvec = cv2.solvePnP(view['obj'], view['uv'], K, None, flags=cv2.SOLVEPNP_IPPE)
    T           = np.eye(4)
    T[:3, :3]   = cv2.Rodrigues(rvec)[0]
    T[:3, 3]    = tvec.ravel()
    return T


def look_at(position : np.ndarray, target : np.ndarray, x_ref : np.ndarray, roll : float):
    # camera pose at position looking at target (optical z axis), x as close as possible to x_ref, then turned
    # by roll (rad) about the optical axis
    z           = (target - position) / np.linalg.norm(target - position)
    x           = x_ref - (x_ref @ z) * z
    x          /= np.linalg.norm(x)
    T           = np.eye(4)
    T[:3, :3]   = R.from_rotvec(roll * z).as_matrix() @ np.column_stack((x, np.cross(z, x), z))
    T[:3, 3]    = position
    return T


def plan_views(T_cam : np.ndarray, center : np.ndarray, args):
    # camera poses looking at the board center: from the start direction with each distinct roll, then tilted
    # by each of tilts away from it at n_azimuths azimuths in turn (neighbors are close, so moves stay short),
    # each at the start distance times one of dist_scales and turned by one of rolls, cycling through both lists
    back    = T_cam[:3, 3] - center
    dist    = np.linalg.norm(back)
    back   /= dist
    side    = T_cam[:3, 0] - (T_cam[:3, 0] @ back) * back       # the tilt axis at azimuth 0
    side   /= np.linalg.norm(side)
    views   = [(0.0, 0.0, roll, 1.0) for roll in sorted(set(args['rolls']))]
    for tilt in args['tilts']:
        for az in np.linspace(0, 2 * np.pi, args['n_azimuths'], endpoint=False):
            k   = len(views)
            views.append((tilt, az, args['rolls'][k % len(args['rolls'])], args['dist_scales'][k % len(args['dist_scales'])]))
    poses   = []
    for tilt, az, roll, scale in views:
        axis        = R.from_rotvec(az * back).apply(side)
        direction   = R.from_rotvec(np.radians(tilt) * axis).apply(back)
        poses.append(look_at(center + scale * dist * direction, center, T_cam[:3, 0], np.radians(roll)))
    return poses


def path_problems(q_from : np.ndarray, q_to : np.ndarray, branch : tuple, arm : dict, args):
    # what keeps the joint move from q_from to q_to from being made, empty when nothing: the table every
    # step_angle on the way, everything (clearance.problems: reach, joint limits, IK branch, singularities,
    # the table) at its end
    if q_to is None:
        return ['out of reach']
    for q in joint_path(q_from, q_to, args['step_angle'])[:-1]:
        if table_problem(q, arm, args):
            return table_problem(q, arm, args)
    return problems(q_to, branch, arm, args['run_margins'], args)


def reachable(robot, arm : dict, T_cams : list, X : np.ndarray, q : np.ndarray, args):
    # joint angles for each view's flange pose (with the camera at X), each solved next to the last reachable one;
    # views whose joint move from the last one fails path_problems (the nearest IK solution can be on another
    # branch after a view is left out) or turns a joint more than max_joint_turn are left out, and the move back
    # to the start must pass too
    qs, q_start = [], q
    branch      = ur_branch(q)
    for i, T in enumerate(T_cams):
        q_new   = robot.ik(matrix_to_pose(T @ np.linalg.inv(X)), q)
        why     = path_problems(q, q_new, branch, arm, args)
        if not why and np.abs(q_new - q).max() > args['max_joint_turn']:
            why = [f"joint {np.argmax(np.abs(q_new - q)) + 1} would turn {np.degrees(np.abs(q_new - q).max()):.0f} deg"]
        if why:
            print(f"view {i} left out: {', '.join(why)}")
            continue
        qs.append(q_new)
        q = q_new
    why = path_problems(q, q_start, branch, arm, args)
    if why:
        raise SystemExit(f"the move back to the start fails ({', '.join(why)}), start from a higher pose")
    return qs


def collect(robot, vision, detector, board, qs : list, q_start : np.ndarray, args):
    # visit every view, let the arm settle and keep the flange pose with the board corners seen there, then go
    # back to the start; a failed move ends it with the views so far
    views = []
    try:
        for i, q in enumerate(qs):
            robot.move_j(q, args['move_speed'], args['move_acc'])
            time.sleep(args['settle'])
            view = capture(vision, detector, board, args)
            if view is None:
                print(f"view {i + 1}/{len(qs)}: board not found, skipped")
                continue
            view['T'], view['q'] = pose_to_matrix(robot.get_tcp_pose()), robot.get_q()
            views.append(view)
            print(f"view {i + 1}/{len(qs)}: {len(view['ids'])} corners")
        robot.move_j(q_start, args['move_speed'], args['move_acc'])
    except RuntimeError as error:
        print(f"{error}, stopping with {len(views)} views: check the arm and the pendant")
    return views


def project(p : np.ndarray, T_flanges : np.ndarray, obj : np.ndarray, index : np.ndarray, K : np.ndarray):
    # pixels of the board points for the camera mount X = p[:6] (flange frame) and the board pose Y = p[6:]
    # (UR base frame), both [x, y, z, rx, ry, rz]: camera <- flange <- base <- board, per point's view index
    T   = np.linalg.inv(pose_to_matrix(p[:6])) @ np.linalg.inv(T_flanges) @ pose_to_matrix(p[6:])
    pc  = np.einsum('nij,nj->ni', T[index, :3, :3], obj) + T[index, :3, 3]
    return pc[:, :2] / pc[:, 2:] * K[[0, 1], [0, 1]] + K[:2, 2]


def solve(views : list, K : np.ndarray, X0 : np.ndarray):
    # least squares over every corner of every view, from the CAD mount and the board where the first view puts
    # it; also the 1-sigma spread of the mount from the pixel residuals alone (robot and intrinsics errors are
    # not in it, so the true uncertainty is larger)
    T_flanges   = np.array([v['T'] for v in views])
    uv          = np.vstack([v['uv'] for v in views])
    obj         = np.vstack([v['obj'] for v in views])
    index       = np.concatenate([np.full(len(v['uv']), i) for i, v in enumerate(views)])
    Y0          = T_flanges[0] @ X0 @ board_in_camera(views[0], K)
    fit         = least_squares(lambda p: (project(p, T_flanges, obj, index, K) - uv).ravel(),
                                np.concatenate((matrix_to_pose(X0), matrix_to_pose(Y0))), method='lm')
    r           = fit.fun.reshape(-1, 2)
    sigma2      = (fit.fun @ fit.fun) / (fit.fun.size - fit.x.size)
    std         = np.sqrt(np.diag(np.linalg.inv(fit.jac.T @ fit.jac) * sigma2))[:6]
    per_view    = [np.sqrt(np.mean(np.sum(r[index == i] ** 2, axis=1))) for i in range(len(views))]
    return {'X': pose_to_matrix(fit.x[:6]), 'Y': pose_to_matrix(fit.x[6:]), 'rms': np.sqrt(np.mean(np.sum(r ** 2, axis=1))),
            'per_view': np.array(per_view), 'std': std}


def difference(A : np.ndarray, B : np.ndarray):
    # how far apart two poses are: translation (mm) and rotation (deg)
    return np.linalg.norm(A[:3, 3] - B[:3, 3]) * 1000, np.degrees(np.linalg.norm(R.from_matrix(A[:3, :3] @ B[:3, :3].T).as_rotvec()))


def report(result : dict, args):
    # the fitted mount to paste into cam_in_flange, how well it fits, and how far it is from the CAD
    # (and, in sim, from the camera's true mount)
    X       = result['X']
    print(f"cam_in_flange = {np.round(matrix_to_pose(X), 4).tolist()}")
    print(f"reprojection rms {result['rms']:.3f} px over {len(result['per_view'])} views, per view: "
          f"{np.round(result['per_view'], 2).tolist()}")
    print(f"1-sigma from pixel noise: position {np.round(result['std'][:3] * 1000, 2).tolist()} mm, "
          f"rotation {np.round(np.degrees(result['std'][3:]), 3).tolist()} deg")
    print("from the CAD mount: {:.1f} mm, {:.2f} deg".format(*difference(X, pose_to_matrix(args['cam_in_flange']))))
    if args['sim']:
        print("from the sim camera's true mount: {:.2f} mm, {:.3f} deg".format(*difference(X, pose_to_matrix(args['sim_cam_in_flange']))))


def save(path : Path, views : list, K : np.ndarray, result : dict):
    # every view (flange pose, joint angles, corner pixels and ids, image) and the fit, enough to fit again
    # without the robot
    np.savez_compressed(path, K=K, T=np.array([v['T'] for v in views]), q=np.array([v['q'] for v in views]),
                        images=np.array([v['color'] for v in views]),
                        uv=np.vstack([v['uv'] for v in views]), ids=np.concatenate([v['ids'] for v in views]),
                        index=np.concatenate([np.full(len(v['ids']), i) for i, v in enumerate(views)]),
                        X=result['X'], Y=result['Y'])


def load(path : Path, board):
    # the views of a run saved by save(), board points from the ids (so a corrected square_len applies)
    d       = np.load(path)
    corners = board.getChessboardCorners()
    views   = [{'T': d['T'][i], 'uv': d['uv'][d['index'] == i], 'ids': d['ids'][d['index'] == i],
                'obj': corners[d['ids'][d['index'] == i]]} for i in range(len(d['T']))]
    return views, d['K']


def main(args):
    board   = make_board(args)
    if args['print_board']:
        save_board_pdf(board, args)
        return
    X0      = pose_to_matrix(args['cam_in_flange'])

    # fit a saved run again (e.g. with a measured square_len)
    if args['refit']:
        views, K = load(Path(args['refit']), board)
        report(solve(views, K, X0), args)
        return

    vision      = Vision(args)
    detector    = cv2.aruco.CharucoDetector(board)
    K           = camera_matrix(vision.intrinsics)
    robot       = URRobot(args)
    robot.set_tcp([0, 0, 0, 0, 0, 0])       # every pose below is the flange's
    arm         = load_arm(args)            # the arm's shapes, to keep it off the table

    try:
        # the start view: the camera must see the board, which with the CAD mount places the board's center
        if args['start_q'] is not None:
            robot.move_j(args['start_q'], args['move_speed'], args['move_acc'])
        q_start     = robot.get_q()
        start       = capture(vision, detector, board, args)
        if start is None:
            raise SystemExit("the camera does not see the board, move the arm so it does and start again")
        T_cam       = pose_to_matrix(robot.get_tcp_pose()) @ X0
        cols, rows  = args['board_squares']
        center      = (T_cam @ board_in_camera(start, K) @ [cols * args['square_len'] / 2, rows * args['square_len'] / 2, 0, 1])[:3]
        print(f"board center {np.round(center * 1000)} mm, {np.linalg.norm(T_cam[:3, 3] - center) * 1000:.0f} mm from the camera")

        # every view checked for reach before the arm moves, then visited
        qs = reachable(robot, arm, plan_views(T_cam, center, args), X0, q_start, args)
        if not args['sim']:
            input(f"{len(qs)} views, the arm moves at {args['move_speed']} rad/s around the board: "
                  "press Enter to start (Ctrl+C to stop) ")
        views = collect(robot, vision, detector, board, qs, q_start, args)
    finally:
        robot.close()
        vision.close()
        cv2.destroyAllWindows()

    # fit, keep the data, report
    if len(views) < args['min_views']:
        raise SystemExit(f"only {len(views)} views with the board, need {args['min_views']}")
    result  = solve(views, K, X0)
    out     = Path(args['out_dir'])
    out.mkdir(exist_ok=True)
    path    = out / f"hand_eye_{time.strftime('%Y%m%d_%H%M%S')}.npz"
    save(path, views, K, result)
    print(f"saved {path}")
    report(result, args)


if __name__ == "__main__":

    args = {}

    # real arm + RealSense, or URSim (docker, ports on localhost) + MuJoCo camera and board
    args['sim']             = False
    args['robot_ip']        = "127.0.0.1" if args['sim'] else "192.168.1.100"

    # the CAD camera mount (flange frame [x, y, z, rx, ry, rz]), where the fit starts and the views are planned
    # from; in sim the camera really sits at sim_cam_in_flange (the touch estimate of 2026-10-05, 5.5 deg / 10 mm
    # from the CAD), which the fit has to find
    args['cam_in_flange']   = [0.0325, 0.0911, 0.0290, 0.0, 2.2214, 2.2214]
    args['sim_cam_in_flange'] = [0.0248, 0.0969, 0.0264, 0.0344, 2.109, 2.3134] if args['sim'] else None
    args['width']           = 640
    args['height']          = 480
    args['fps']             = 30

    # ChArUco board: squares (columns, rows), square and marker size (m, set square_len to the printed board's
    # measured size, marker_len in proportion), ArUco dictionary; printing on paper_mm (US Letter landscape,
    # A4 is [297, 210]), print_board = a pdf path writes the board there and stops
    args['board_squares']   = (8, 6)
    args['square_len']      = 0.030
    args['marker_len']      = 0.022
    args['board_dict']      = 'DICT_4X4_50'
    args['print_board']     = None
    args['paper_mm']        = [279.4, 215.9]
    args['print_px_per_square'] = 300

    # simulated scene, same as visual_servo.py, plus the board standing upright 0.45 m in front of the camera at
    # home_q (board frame: origin at its top-left corner, x right, y down, z into the board), with a white margin
    args['xml_file']        = 'ur10e.xml'
    args['sim_fovy']        = 43.1
    args['beam_pose']       = [-0.012, 1.562, 0.161, 0, 0, 0]
    args['beam_size']       = [0.224, 1.5, 0.263]
    args['beam_rgba']       = [0.8, 0.5, 0.2, 1]
    args['board_pose']      = [0.015, 0.85, 0.48, -np.pi / 2, 0, 0] if args['sim'] else None
    args['board_margin']    = 0.01
    args['render']          = True
    args['cam_lookat']      = [0.1, 0.9, 0.5]
    args['cam_azi']         = 150
    args['cam_ele']         = -25
    args['cam_dist']        = 3.0

    # start: the arm moves to start_q first (visual_servo.py's home, sim), or starts wherever it is (None: put the
    # camera ~0.4 m from the board, looking at it, with the pendant or freedrive)
    args['start_q']         = [-1.5593, -1.4434, -2.1595, 2.0374, -1.5729, -0.0185] if args['sim'] else None

    # views around the board center (deg): the start view with every roll, then tilts at n_azimuths azimuths,
    # cycling through rolls about the optical axis (30 deg apart from view to view, to keep the wrist's moves
    # short) and distances (times the start distance)
    args['tilts']           = [15, 30]
    args['n_azimuths']      = 6
    args['rolls']           = [-30, 0, 30, 0]
    args['dist_scales']     = [0.85, 1.0, 1.15]

    # motion: joint moves (rad/s, rad/s^2), still time before each image (s), frames dropped to get a fresh one;
    # no step of a move (every step_angle) closer to a singularity than run_margins (see visual_servo.py) or with
    # a part of the arm or tool closer to the table than table_margin (table top at table_z in UR's base frame,
    # 0 = the base sits on it), and no joint turning more than max_joint_turn from the view before
    args['move_speed']      = 0.3
    args['move_acc']        = 0.5
    args['settle']          = 0.5
    args['flush_frames']    = 5
    args['run_margins']     = {'shoulder': 0.05, 'elbow': 0.10, 'wrist': 0.10}
    args['table_z']         = 0.0
    args['table_margin']    = 0.05
    args['step_angle']      = np.radians(2)
    args['max_joint_turn']  = np.radians(135)

    # detection and fit: corners needed in a view, views needed in all
    args['min_corners']     = 8
    args['min_views']       = 8

    # output, or set refit to a saved results/hand_eye_*.npz to fit it again without the robot
    args['out_dir']         = 'results'
    args['refit']           = None

    main(args)
