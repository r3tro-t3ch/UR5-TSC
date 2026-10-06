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
# Measure the end of a wooden beam by touch, after visual_servo.py has left the
# camera looking straight at the beam's end face. The depth camera gives a
# first estimate of the face. Then the edge finder's two L-shaped plates press
# on the faces at the top, right and left edges: the front plate on the end
# face, the top plate on the side face. Each press moves in until the first ball
# touches (the robot's own contact detection), then pushes in force mode while
# free to tilt, so the plate settles with all three balls on the wood. Planes
# through the ball contacts give the width and the angles between the faces;
# they are compared with the camera's plane and plotted with matplotlib.

import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from env.ur_robot import URRobot
from env.vision import Vision
from utils.utils import pose_to_matrix, matrix_to_pose, transform_points
from visual_servo import find_wood_face, near_singularity

# edge finder balls in the flange frame (m), fitted to edge_finder.stl: each plate has three r = 10 mm balls
# on a 100 mm equilateral triangle, pressing along the plate's outward normal
BALL_R  = 0.010
PLATES  = {'front':  {'balls': np.array([[0.050, -0.0058, 0.014], [-0.050, -0.0058, 0.014], [0.0, -0.0924, 0.014]]),
                      'normal': np.array([0.0, 0.0, 1.0])},
           'top':    {'balls': np.array([[0.050, 0.0442, 0.064], [-0.050, 0.0442, 0.064], [0.0, 0.0442, 0.1506]]),
                      'normal': np.array([0.0, -1.0, 0.0])}}

# beam frame from the camera: origin at the end face center, x left, y up, z into the beam (as seen by the robot);
# outward face normals, and the side of the end face where the top plate goes at each edge
NORMALS = {'front': [0, 0, -1], 'top': [0, 1, 0], 'right': [-1, 0, 0], 'left': [1, 0, 0]}

# the presses in order: (edge, plate), the front plate always on the end face, the top plate on that edge's side
# face. The side face goes first at every edge: the shelf reaches 160 mm along the tool, so the end face press
# needs the real side face to keep it clear (the camera's guess can be centimeters off)
ROUTINE = [('top', 'top'), ('top', 'front'), ('right', 'top'), ('right', 'front'), ('left', 'top'), ('left', 'front')]


def fit_plane(points : np.ndarray):
    # least squares plane: centroid and unit normal (the direction of least spread)
    centroid = points.mean(axis=0)
    return centroid, np.linalg.svd(points - centroid, full_matrices=False)[2][2]


def angle(a : np.ndarray, b : np.ndarray):
    # angle between two unit vectors (deg)
    return np.degrees(np.arccos(np.clip(a @ b, -1.0, 1.0)))


def camera_face(vision, robot, args, rng):
    # wood points of the end face from a few depth frames, in UR's base frame (the TCP is the flange here),
    # and the flange pose they were taken from
    T_base_cam  = pose_to_matrix(robot.get_tcp_pose()) @ pose_to_matrix(args['cam_in_flange'])
    points      = []
    for _ in range(args['n_frames']):
        face = find_wood_face(*vision.get_frames(), vision.intrinsics, args, rng)
        if face is not None:
            points.append(face['points'])
    if not points:
        raise SystemExit("the camera sees no wood face, run visual_servo.py first")
    return transform_points(T_base_cam, np.vstack(points)), pose_to_matrix(robot.get_tcp_pose())


def beam_frame(points : np.ndarray):
    # beam frame on the camera's end face (outward normal towards the robot base) and the face's half sizes
    center, n   = fit_plane(points)
    z           = n if n @ center > 0 else -n               # into the beam, away from the base
    y           = np.array([0, 0, 1.0]) - z[2] * z          # up (UR base z) within the face
    y          /= np.linalg.norm(y)
    x           = np.cross(y, z)

    # face extent along x and y, ignoring the outermost 0.1 % of points (flying pixels at the edges)
    u, v        = (points - center) @ x, (points - center) @ y
    lo, hi      = np.percentile(np.c_[u, v], [0.1, 99.9], axis=0)
    T           = np.eye(4)
    T[:3, :3]   = np.column_stack((x, y, z))
    T[:3, 3]    = center + x * (lo[0] + hi[0]) / 2 + y * (lo[1] + hi[1]) / 2
    return T, {'top': (hi[1] - lo[1]) / 2, 'right': (hi[0] - lo[0]) / 2, 'left': (hi[0] - lo[0]) / 2}


def reach(plate : str):
    # how far the plate's ball tips are from the flange along the plate's normal (m, < 0 behind the flange)
    return PLATES[plate]['balls'][0] @ PLATES[plate]['normal'] + BALL_R


def camera_planes(T_beam : np.ndarray, half : dict):
    # first guess of the four faces from the camera: a point on each and its outward normal (UR base frame)
    return {face: (T_beam[:3, 3] + T_beam[:3, :3] @ (np.array(NORMALS[face], float) * half.get(face, 0.0)),
                   T_beam[:3, :3] @ np.array(NORMALS[face], float)) for face in NORMALS}


def edge_pose(planes : dict, center : np.ndarray, edge : str):
    # flange pose at one edge of the end face: tool axis into the end face, top plate over the edge's side face,
    # both plates' ball tips on the current estimate of the two faces, centered on the edge at the face center
    (p_end, n_end), (p_side, n_side) = planes['front'], planes[edge]
    z           = -n_end
    y           = n_side - (n_side @ z) * z
    y          /= np.linalg.norm(y)
    x           = np.cross(y, z)                        # along the edge
    T           = np.eye(4)
    T[:3, :3]   = np.column_stack((x, y, z))
    T[:3, 3]    = np.linalg.solve(np.vstack((n_end, n_side, x)),
                                  [n_end @ p_end + reach('front'), n_side @ p_side + reach('top'), x @ center])
    return T


def teach_top_edge(robot, planes : dict, center : np.ndarray, args):
    # correct the camera's guess by hand: in freedrive the tool is seated on the top edge (front plate balls on
    # the end face, top plate balls on the top face), and the camera's whole estimate is moved rigidly onto it,
    # keeping the camera's position along the edge. Then the tool backs off up and away. Returns the corrected
    # faces and center, and the correction (4x4, UR base frame)
    robot.free_drive(True)
    input("freedrive: seat the tool on the top edge of the end face (front plate balls on the end face, "
          "top plate balls on the top face), let go, then press Enter ")
    robot.free_drive(False)
    T_taught        = pose_to_matrix(robot.get_tcp_pose())
    T_guess         = edge_pose(planes, center, 'top')
    T_taught[:3, 3] += ((T_guess[:3, 3] - T_taught[:3, 3]) @ T_taught[:3, 0]) * T_taught[:3, 0]
    fix             = T_taught @ np.linalg.inv(T_guess)

    # off the wood: first up and back a little (no sliding on the top face), then straight back
    T_off           = pose_to_matrix(robot.get_tcp_pose())
    T_off[:3, 3]   += args['clear'] * (T_off[:3, 1] - T_off[:3, 2])
    robot.move_l(matrix_to_pose(T_off), args['touch_speed'], args['touch_acc'])
    robot.move_l(matrix_to_pose(retracted(T_off, args)), args['lin_speed'], args['lin_acc'])
    return ({face: (transform_points(fix, p[None])[0], fix[:3, :3] @ n) for face, (p, n) in planes.items()},
            transform_points(fix, center[None])[0], fix)


def plate_tcp(plate : str):
    # TCP for pressing with a plate, in the flange frame: at the center of its three ball tips, z along its normal
    balls, n    = PLATES[plate]['balls'], PLATES[plate]['normal']
    x           = balls[0] - balls.mean(axis=0)
    T           = np.eye(4)
    T[:3, :3]   = np.column_stack((x / np.linalg.norm(x), np.cross(n, x) / np.linalg.norm(x), n))
    T[:3, 3]    = balls.mean(axis=0) + BALL_R * n
    return T


def retracted(T_edge : np.ndarray, args):
    # the edge pose backed off from the end face, for going from edge to edge
    T           = T_edge.copy()
    T[:3, 3]   -= args['retract'] * T_edge[:3, 2]
    return T


def plan_press(planes : dict, measured : set, center : np.ndarray, edge : str, plate : str, args):
    # one press of a plate at an edge, from the current face estimates: start pose (backed off from the face,
    # clear of the other face, less so once that face is measured), push direction, farthest travel, retract
    T_edge          = edge_pose(planes, center, edge)
    other           = 'top' if plate == 'front' else 'front'
    clear           = args['clear_measured'] if (edge if other == 'top' else 'front') in measured else args['clear']
    offset          = np.eye(4)
    offset[:3, 3]   = -args['standoff'] * PLATES[plate]['normal'] - clear * PLATES[other]['normal']
    start           = T_edge @ offset
    direction       = T_edge[:3, :3] @ PLATES[plate]['normal']
    limit           = start.copy()
    limit[:3, 3]   += args['max_travel'] * direction
    return {'edge': edge, 'plate': plate, 'face': 'front' if plate == 'front' else edge,
            'start': start, 'direction': direction, 'limit': limit, 'retract': retracted(T_edge, args)}


def route(prev, press : dict, planes : dict, center : np.ndarray, args):
    # retracted poses from the previous press to this one: back from the old edge, through the top edge when going
    # from side to side (the wrist turns back 90 deg and on 90 deg instead of 180 deg past its limit), to the new
    # edge. The first one is reached by a joint move
    if prev is None:
        return [press['retract']]
    if prev['edge'] == press['edge']:
        return []
    via = [] if 'top' in (prev['edge'], press['edge']) else [retracted(edge_pose(planes, center, 'top'), args)]
    return [prev['retract']] + via + [press['retract']]


def check_reach(robot, poses : list, q : np.ndarray, args, first_by_joint_move : bool):
    # follow the joints along a list of poses: each needs an IK solution clear of the singularities, and between
    # poses no joint may turn more than max_joint_turn (the largest planned turn is the wrist's 90 deg between
    # edges; more means the controller's solution went the other way round because a straight line move would
    # take a wrist past its limit). Raises RuntimeError, returns the last joint angles
    for i, T in enumerate(poses):
        q_new = robot.ik(matrix_to_pose(T), q)
        near  = near_singularity(q_new, args['run_margins']) if q_new is not None else ['out of reach']
        turn  = np.abs(q_new - q).max() if q_new is not None else 0.0
        if not near and turn > args['max_joint_turn'] and not (first_by_joint_move and i == 0):
            near = [f"joint {np.argmax(np.abs(q_new - q)) + 1} would turn {np.degrees(turn):.0f} deg"]
        if near:
            raise RuntimeError(f"pose {i} of the route: {', '.join(near)}")
        q = q_new
    return q


def check_plan(robot, planes : dict, center : np.ndarray, args):
    # before moving: the whole routine as planned from the camera alone must be reachable
    poses, prev = [], None
    for edge, plate in ROUTINE:
        press   = plan_press(planes, set(), center, edge, plate, args)
        poses  += route(prev, press, planes, center, args) + [press['start'], press['limit'], press['start']]
        prev    = press
    check_reach(robot, poses + [prev['retract']], robot.get_q(), args, True)


def aligned(T : np.ndarray, plate : str, normal : np.ndarray):
    # flange pose T turned about the plate's ball center so the plate faces a plane with this outward normal
    push        = T[:3, :3] @ PLATES[plate]['normal']
    turn        = R.align_vectors([-normal], [push])[0].as_matrix()
    center      = transform_points(T, PLATES[plate]['balls'].mean(axis=0)[None])[0]
    T_new       = np.eye(4)
    T_new[:3, :3] = turn @ T[:3, :3]
    T_new[:3, 3]  = center + turn @ (T[:3, 3] - center)
    return T_new


def ball_forces(T : np.ndarray, wrench : np.ndarray, plate : str):
    # each ball's push on the wood (N) from the wrist wrench at the plate TCP (flange pose T): the force along
    # the push and the torques about the two in-plane axes give three equations for the three balls
    tcp     = T @ plate_tcp(plate)
    tips    = (PLATES[plate]['balls'] + BALL_R * PLATES[plate]['normal'] - plate_tcp(plate)[:3, 3]) @ T[:3, :3].T
    lever   = np.cross(tips, tcp[:3, 2])                    # torque about the TCP per newton, one row per ball
    M       = np.vstack((np.ones(3), lever @ tcp[:3, 0], lever @ tcp[:3, 1]))
    f       = np.linalg.solve(M, [wrench[:3] @ tcp[:3, 2], wrench[3:] @ tcp[:3, 0], wrench[3:] @ tcp[:3, 1]])
    return f if f.sum() > 0 else -f     # the sensor's sign convention does not matter, the balls push


def press_face(robot, sim_env, press : dict, args):
    # move in until the first ball touches, then press flat so all three balls rest on the face; records the
    # flange pose, the three ball centers, each ball's force and whether the arm came to rest; False on a miss
    balls = PLATES[press['plate']]['balls']
    robot.move_l(matrix_to_pose(press['start']), args['lin_speed'], args['lin_acc'])
    if sim_env is None:
        robot.zero_ft()
        if not robot.move_until_contact(matrix_to_pose(press['limit']), args['touch_speed'], args['touch_acc']):
            return False
        # press about the balls' center, then the flange is the TCP again
        tcp = plate_tcp(press['plate'])
        robot.set_tcp(matrix_to_pose(tcp))
        poses, wrenches = robot.press(args['press_force'], args['press_time'], args['press_damping'],
                                      args['press_limits'])
        robot.set_tcp([0, 0, 0, 0, 0, 0])
        T       = pose_to_matrix(poses.mean(axis=0)) @ np.linalg.inv(tcp)
        forces  = ball_forces(T, wrenches.mean(axis=0), press['plate'])
        # how much the plate still moved: TCP travel, and its tilt seen at the balls (58 mm from the TCP)
        moving  = max(np.ptp(poses[:, :3], axis=0).max(), np.ptp(poses[:, 3:], axis=0).max() * 0.058)
    else:
        # URSim has no contact or force: find the face the first ball meets and lay the plate flush on it
        hit = sim_env.first_contact(transform_points(press['start'], balls), BALL_R, press['direction'])
        if hit is None or hit[0] > args['max_travel']:
            return False
        T       = aligned(press['start'], press['plate'], hit[2])
        T[:3, 3] -= (np.mean((transform_points(T, balls) - hit[1]) @ hit[2]) - BALL_R) * hit[2]
        robot.move_l(matrix_to_pose(T), args['touch_speed'], args['touch_acc'])
        T       = pose_to_matrix(robot.get_tcp_pose())
        forces  = np.full(3, args['press_force'] / 3)
        moving  = 0.0
    press.update(T=T, centers=transform_points(T, balls), forces=forces, moving=moving,
                 flat=bool(moving < args['still_tol'] and forces.min() > args['min_ball_force']))
    robot.move_l(matrix_to_pose(press['start']), args['lin_speed'], args['lin_acc'])
    return True


def run(robot, sim_env, planes : dict, center : np.ndarray, args):
    # press edge by edge in ROUTINE order: each press is planned from the latest face estimates and its measured
    # plane replaces the camera's guess of that face, then back to where the arm started; stops at the first
    # miss or failed move and returns the presses that touched
    q_home  = robot.get_q()
    done    = []
    prev    = None
    try:
        for edge, plate in ROUTINE:
            press   = plan_press(planes, {p['face'] for p in done}, center, edge, plate, args)
            path    = route(prev, press, planes, center, args)
            check_reach(robot, path + [press['start'], press['limit']], robot.get_q(), args, prev is None)
            if prev is None:
                robot.move_j(robot.ik(matrix_to_pose(path[0]), q_home), args['move_speed'], args['move_acc'])
                path = path[1:]
            for T in path:
                robot.move_l(matrix_to_pose(T), args['lin_speed'], args['lin_acc'])
            if not press_face(robot, sim_env, press, args):
                print(f"no contact within {args['max_travel'] * 1000:.0f} mm: {edge} edge, {plate} plate, stopping")
                break
            done.append(press)
            prev = press
            planes.update({face: (plane['point'], plane['normal']) for face, plane in fit_faces(done).items()})
            print(f"{edge:5s} edge  {plate:5s} plate on the {press['face']:5s} face: "
                  f"ball forces {np.round(press['forces'], 1)} N, still within {press['moving'] * 1000:.3f} mm, "
                  f"{'all three balls on the wood' if press['flat'] else 'NOT SURE all three balls touched'}")
        robot.move_l(matrix_to_pose(press['retract']), args['lin_speed'], args['lin_acc'])
        robot.move_j(q_home, args['move_speed'], args['move_acc'])
    except RuntimeError as error:
        print(f"{error}, stopping here: check the arm and the pendant")
    return done


def fit_faces(presses : list):
    # per face: plane through all its ball centers moved one radius into the wood (point, outward normal,
    # contacts, rms of the contacts about it), and whether every press on it lay flat
    planes = {}
    for face in NORMALS:
        on_face = [press for press in presses if press['face'] == face]
        if not on_face:
            continue
        centers     = np.vstack([press['centers'] for press in on_face])
        point, n    = fit_plane(centers)
        n           = n if n @ on_face[0]['direction'] < 0 else -n
        contacts    = centers - BALL_R * n
        planes[face] = {'point': point - BALL_R * n, 'normal': n, 'contacts': contacts,
                        'rms': np.sqrt(np.mean(((contacts - point + BALL_R * n) @ n) ** 2)),
                        'flat': all(press['flat'] for press in on_face),
                        'min_force': min(press['forces'].min() for press in on_face)}
    return planes


def measure(planes : dict, cam_points : np.ndarray, half : dict):
    # sizes (mm) and angles between faces (deg, inside the beam, 90 for a square cut) from the touch planes that
    # exist, and the camera's end face against the touched one
    n       = {face: plane['normal'] for face, plane in planes.items()}
    name    = {'front': 'end', 'top': 'top', 'right': 'right', 'left': 'left'}
    results = {'width, camera (mm)': 2 * half['left'] * 1000, 'height, camera (mm)': 2 * half['top'] * 1000}
    if 'left' in n and 'right' in n:
        across = (n['left'] - n['right']) / np.linalg.norm(n['left'] - n['right'])
        results['width, touch (mm)'] = (planes['left']['point'] - planes['right']['point']) @ across * 1000
        results['left / right, not parallel by (deg)'] = angle(n['left'], -n['right'])
    for a, b in [('front', 'top'), ('front', 'right'), ('front', 'left'), ('top', 'right'), ('top', 'left')]:
        if a in n and b in n:
            results[f"{name[a]} / {name[b]} (deg)"] = 180 - angle(n[a], n[b])
    if 'front' in n:
        cam_c, cam_n = fit_plane(cam_points)
        cam_n   = cam_n if cam_n @ n['front'] > 0 else -cam_n
        results['end face flatness, rms (mm)'] = planes['front']['rms'] * 1000
        results['camera vs touch, end face tilt (deg)'] = angle(cam_n, n['front'])
        results['camera vs touch, end face offset (mm)'] = (cam_c - planes['front']['point']) @ n['front'] * 1000
    return results


def plot(planes : dict, cam_points : np.ndarray, results : dict, half : dict, args):
    # the touched planes as patches over the beam's end, the ball contacts and the camera's points, seen from
    # the robot in a frame on the touched end face (x right, y into the beam, z up), the angles on the edges
    # and every number alongside
    y       = -planes['front']['normal']                                    # into the beam
    z       = planes['top']['normal'] - (planes['top']['normal'] @ y) * y   # up
    z      /= np.linalg.norm(z)
    T       = np.eye(4)
    T[:3, :3] = np.column_stack((np.cross(y, z), y, z))
    T[:3, 3]  = planes['front']['point']
    T_inv   = np.linalg.inv(T)

    # patch extents (m): the face's width and height, and how far into the beam the top and sides are drawn
    w, h, d = half['left'], half['top'], args['plot_depth']
    extents = {'front': [(-w, w), None, (-h, h)], 'top': [(-w, w), (0, d), None],
               'right': [None, (0, d), (-h, h)], 'left': [None, (0, d), (-h, h)]}
    colors  = {'front': 'tab:orange', 'top': 'tab:green', 'right': 'tab:blue', 'left': 'tab:purple'}

    fig     = plt.figure(figsize=(15, 8))
    grid    = fig.add_gridspec(1, 2, width_ratios=(1.6, 1))
    ax      = fig.add_subplot(grid[0], projection='3d')
    for face, plane in planes.items():
        # solve the plane for its missing coordinate over the corners of the other two
        p0, n   = transform_points(T_inv, plane['point'][None])[0], T_inv[:3, :3] @ plane['normal']
        free    = [i for i in range(3) if extents[face][i] is not None]
        dep     = extents[face].index(None)
        a, b    = np.meshgrid(extents[face][free[0]], extents[face][free[1]])
        corners = [None] * 3
        corners[free[0]], corners[free[1]] = a, b
        corners[dep] = p0[dep] - (n[free[0]] * (a - p0[free[0]]) + n[free[1]] * (b - p0[free[1]])) / n[dep]
        ax.plot_surface(*[c * 1000 for c in corners], color=colors[face], alpha=0.3)
        c = transform_points(T_inv, plane['contacts']) * 1000
        ax.scatter(*c.T, color=colors[face], s=40, depthshade=False, label=f"{face} face, {len(c)} ball contacts")

    cam = transform_points(T_inv, cam_points[np.random.default_rng(0).choice(len(cam_points), 3000)]) * 1000
    ax.scatter(*cam.T, color='0.4', s=1, alpha=0.3, label='depth camera, end face')

    # angles on the edges they belong to, the width under the end face
    W, H, D = w * 1000, h * 1000, d * 1000
    for key, label, pos in [('end / top (deg)', 'end/top {:.2f}°', (0, 0, H)),
                            ('end / right (deg)', 'end/right {:.2f}°', (W, 0, 0)),
                            ('end / left (deg)', 'end/left {:.2f}°', (-W, 0, 0)),
                            ('top / right (deg)', 'top/right {:.2f}°', (W, D / 2, H)),
                            ('top / left (deg)', 'top/left {:.2f}°', (-W, D / 2, H)),
                            ('width, touch (mm)', 'width {:.2f} mm', (0, 0, -H - 30))]:
        if key in results:
            ax.text(*pos, label.format(results[key]), fontsize=9, ha='center', fontweight='bold')
    ax.set_xlabel('x, right (mm)'), ax.set_ylabel('y, into the beam (mm)'), ax.set_zlabel('z, up (mm)')
    ax.set_box_aspect((2 * W, D, 2 * H + 60))
    ax.view_init(elev=22, azim=-62)
    ax.legend(loc='upper left', fontsize=8)
    ax.set_title('beam end, planes through the ball contacts', pad=0)

    # every number, and whether every press lay flat
    lines = [f"{k:40s} {v:8.2f}" for k, v in results.items()]
    lines += ['', 'all 3 balls on the wood (plate still, each ball pushing):']
    lines += [f"  {face:6s} {'yes     ' if p['flat'] else 'NOT SURE'}  weakest ball {p['min_force']:5.1f} N"
              for face, p in planes.items()]
    text    = fig.add_subplot(grid[1])
    text.axis('off')
    text.text(0, 0.5, '\n'.join(lines), family='monospace', fontsize=9, va='center')
    fig.subplots_adjust(left=0, right=0.98, top=0.95, bottom=0.03, wspace=0.05)
    return fig


def save(path : Path, cam_points : np.ndarray, T_beam : np.ndarray, half : dict, presses : list,
         T_capture : np.ndarray, cam_fixed):
    # everything needed to fit and plot again without the robot, plus where the camera was and its corrected
    # mounting when the first edge was taught (nan otherwise)
    np.savez(path, cam_points=cam_points, T_beam=T_beam, half=[half['top'], half['left']], T_capture=T_capture,
             cam_fixed=np.full(6, np.nan) if cam_fixed is None else cam_fixed,
             **{key: np.array([press[key] for press in presses])
                for key in ('edge', 'plate', 'face', 'direction', 'T', 'centers', 'forces', 'moving', 'flat')})


def load(path : Path):
    # a run saved by save()
    d       = np.load(path)
    keys    = ('edge', 'plate', 'face', 'direction', 'T', 'centers', 'forces', 'moving', 'flat')
    presses = [dict(zip(keys, values)) for values in zip(*(d[key] for key in keys))]
    return d['cam_points'], d['T_beam'], {'top': d['half'][0], 'right': d['half'][1], 'left': d['half'][1]}, presses


def report(presses : list, cam_points : np.ndarray, half : dict, path : Path, args):
    # points sorted by face, one plane per face, every measurement they allow, and the plot next to the data
    planes  = fit_faces(presses)
    results = measure(planes, cam_points, half)
    touched = [f"{face} ({len(plane['contacts'])} points)" for face, plane in planes.items()]
    print(f"faces touched: {', '.join(touched)}")
    for k, v in results.items():
        print(f"{k:40s} {v:8.2f}")
    if 'front' in planes and 'top' in planes:
        fig = plot(planes, cam_points, results, half, args)
        fig.savefig(path.with_suffix('.png'), dpi=150)
        print(f"saved {path.with_suffix('.png')}")
        plt.show()


def main(args):
    # fit and plot a saved run again
    if args['replot']:
        cam_points, _, half, presses = load(Path(args['replot']))
        report(presses, cam_points, half, Path(args['replot']), args)
        return

    rng     = np.random.default_rng(0)
    vision  = Vision(args)
    sim_env = vision.sim_env if args['sim'] else None
    robot   = URRobot(args)
    robot.set_tcp([0, 0, 0, 0, 0, 0])   # every pose below is the flange's

    try:
        # the camera's end face sets the beam frame and the face size, which place the presses
        cam_points, T_capture = camera_face(vision, robot, args, rng)
        T_beam, half    = beam_frame(cam_points)
        planes, center  = camera_planes(T_beam, half), T_beam[:3, 3]
        print(f"camera: end face {2 * half['left'] * 1000:.0f} x {2 * half['top'] * 1000:.0f} mm, "
              f"center {np.round(center * 1000)} mm")

        # optionally seat the tool on the top edge by hand: corrects the camera's guess for this run, and gives
        # the camera's mounting (cam_in_flange) that would have seen the face where it really is
        cam_fixed = None
        if args['teach_first']:
            planes, center, fix = teach_top_edge(robot, planes, center, args)
            cam_fixed = matrix_to_pose(np.linalg.inv(T_capture) @ fix @ T_capture @ pose_to_matrix(args['cam_in_flange']))
            moved = np.linalg.norm(transform_points(fix, T_beam[:3, 3][None])[0] - T_beam[:3, 3])
            turned = np.degrees(np.linalg.norm(R.from_matrix(fix[:3, :3]).as_rotvec()))
            print(f"camera guess moved {moved * 1000:.1f} mm and turned {turned:.2f} deg onto the taught edge; "
                  f"cam_in_flange that fits: {np.round(cam_fixed, 4).tolist()}")

        # check the whole routine as planned, then press, correcting the plan as faces are touched
        try:
            check_plan(robot, planes, center, args)
        except RuntimeError as error:
            raise SystemExit(f"not reachable as planned, {error}")
        presses = run(robot, sim_env, planes, center, args)
    finally:
        robot.close()
        vision.close()

    # keep the raw data first, then fit and plot
    out     = Path(args['out_dir'])
    out.mkdir(exist_ok=True)
    path    = out / f"beam_{time.strftime('%Y%m%d_%H%M%S')}.npz"
    save(path, cam_points, T_beam, half, presses, T_capture, cam_fixed)
    print(f"saved {path}")
    report(presses, cam_points, half, path, args)


if __name__ == "__main__":

    args = {}

    # real arm + RealSense, or URSim (docker, ports on localhost) + MuJoCo camera and beam
    args['sim']             = False
    args['robot_ip']        = "127.0.0.1" if args['sim'] else "192.168.1.100"

    # camera color optical frame in the flange frame [x, y, z, rx, ry, rz], same as visual_servo.py
    args['cam_in_flange']   = [0.0325, 0.0911, 0.0290, 0.0, 2.2214, 2.2214]
    args['width']           = 640
    args['height']          = 480
    args['fps']             = 30

    # simulated scene, same as visual_servo.py
    args['xml_file']        = 'ur10e.xml'
    args['sim_fovy']        = 43.1
    args['beam_pose']       = [-0.0741, 1.9744, 0.30, 0, 0, 0.2618]
    args['beam_size']       = [0.19, 1.5, 0.35]
    args['beam_rgba']       = [0.8, 0.5, 0.2, 1]
    args['render']          = True
    args['cam_lookat']      = [0.1, 0.9, 0.5]
    args['cam_azi']         = 150
    args['cam_ele']         = -25
    args['cam_dist']        = 3.0

    # wood face from the depth camera, same as visual_servo.py, stacked over n_frames frames
    args['hsv_low']         = np.array([8, 110, 40])
    args['hsv_high']        = np.array([25, 255, 255])
    args['depth_min']       = 0.3
    args['depth_max']       = 1.5
    args['max_faces']       = 1
    args['plane_tol']       = 0.01
    args['ransac_iters']    = 200
    args['ransac_points']   = 2000
    args['min_pixels']      = 3000
    args['n_frames']        = 10

    # presses (m): the plate starts standoff before the face's estimate and searches up to max_travel for the
    # first contact. A hand-eye error of a few degrees moves the camera's face ~5 cm sideways per 5.7 deg at
    # 0.5 m (its depth stays within ~1.5 cm), so the search covers +-8 cm. The other plate stays clear of its
    # face, less once that face is measured (the first top plate balls land 40 mm - clear behind the end face edge)
    args['standoff']        = 0.08
    args['max_travel']      = 0.16
    args['clear']           = 0.02
    args['clear_measured']  = 0.015
    args['teach_first']     = not args['sim']   # seat the tool on the top edge by hand first (camera not calibrated)
    args['retract']         = 0.20      # back from the end face when going from edge to edge

    # force mode after the first contact: push (N) for press_time (s), tilting freely until all balls touch;
    # limits: [x, y] allowed deviation (m), [z] speed (m/s), [rx, ry] speed (rad/s), [rz] deviation (rad).
    # damping 0..1 (ur_rtde default 0.005), tune on the arm
    args['press_force']     = 20.0
    args['press_time']      = 3.0
    args['press_damping']   = 0.1
    args['press_limits']    = [0.005, 0.005, 0.01, 0.2, 0.2, 0.05]
    args['still_tol']       = 0.0002    # m, the plate moved less than this over the second half of the press
    args['min_ball_force']  = 2.0       # N, every ball pushing at least this much (sensor precision ~3.5 N)

    # speeds: touching (m/s, m/s^2), straight line moves, joint moves (rad/s, rad/s^2)
    args['touch_speed']     = 0.01
    args['touch_acc']       = 0.2
    args['lin_speed']       = 0.1
    args['lin_acc']         = 0.3
    args['move_speed']      = 0.5
    args['move_acc']        = 0.5

    # no pose may come closer to a singularity than this (see visual_servo.py), and no joint may turn more than
    # max_joint_turn between two poses of the route
    args['run_margins']     = {'shoulder': 0.05, 'elbow': 0.10, 'wrist': 0.10}
    args['max_joint_turn']  = np.radians(135)

    # output, or set replot to a saved results/beam_*.npz to fit and plot it again without the robot
    args['out_dir']         = 'results'
    args['replot']          = None
    args['plot_depth']      = 0.2       # m into the beam drawn for the top and side planes

    main(args)
