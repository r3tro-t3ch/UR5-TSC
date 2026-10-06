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
# Couple URSim and MuJoCo for testing without hardware. URSim runs the real UR
# controller; MuJoCo only mirrors its joint angles (no physics) and renders
# color and depth from a camera on the wrist looking at a wood-colored beam,
# optionally on a wood table and with a ChArUco board for hand_eye.py. URSim
# feels no contact, so the beam box also tells how far a ball can move before
# it touches. Optionally show the mirrored arm in a mujoco-python-viewer window.

import mujoco
import multiprocessing
import numpy as np
import signal
from pathlib import Path
from types import SimpleNamespace
from rtde_receive import RTDEReceiveInterface
from scipy.spatial.transform import Rotation as R
from utils.charuco import make_board, board_image
from utils.utils import pose_to_matrix

# checked against the UR10e DH parameters: UR's base frame is the MuJoCo 'base' body turned 180 deg about z,
# and UR's tool flange sits 16.7 mm in front of MuJoCo's 'attachment_site' (same axes)
UR_BASE_IN_MJ_BASE  = pose_to_matrix([0, 0, 0, 0, 0, np.pi])
FLANGE_IN_SITE      = pose_to_matrix([0, 0, 0.0167, 0, 0, 0])

# a MuJoCo camera looks along its -z with y up, the RealSense optical frame along +z with y down
MJ_CAM_IN_OPTICAL   = pose_to_matrix([0, 0, 0, np.pi, 0, 0])


def matrix_to_pos_quat(T : np.ndarray):
    # 4x4 transform -> MuJoCo position and quaternion (w, x, y, z)
    return T[:3, 3], R.from_matrix(T[:3, :3]).as_quat(scalar_first=True)


def build_scene(args):
    # the UR10e model, with an offscreen buffer as big as the camera image
    spec = mujoco.MjSpec.from_file(str(Path(__file__).resolve().parent / args['xml_file']))
    spec.visual.global_.offwidth, spec.visual.global_.offheight = args['width'], args['height']

    # wrist camera at cam_in_flange from UR's flange (the same transform the robot uses as TCP), or where
    # sim_cam_in_flange puts it, to test hand_eye.py on a camera that is not where the CAD says
    site            = spec.site('attachment_site')
    T_wrist_site    = np.eye(4)
    T_wrist_site[:3, :3], T_wrist_site[:3, 3] = R.from_quat(site.quat, scalar_first=True).as_matrix(), site.pos
    T_flange_cam    = pose_to_matrix(args.get('sim_cam_in_flange') or args['cam_in_flange'])
    pos, quat       = matrix_to_pos_quat(T_wrist_site @ FLANGE_IN_SITE @ T_flange_cam @ MJ_CAM_IN_OPTICAL)
    spec.body('wrist_3_link').add_camera(name='wrist_camera', pos=pos, quat=quat, fovy=args['sim_fovy'])

    # the beam, a wood-colored box placed in UR's base frame
    pos, quat       = matrix_to_pos_quat(UR_BASE_IN_MJ_BASE @ pose_to_matrix(args['beam_pose']))
    spec.body('base').add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=np.array(args['beam_size']) / 2,
                               pos=pos, quat=quat, rgba=args['beam_rgba'])

    # the table the robot stands on (optional), a wide slab with its top at table_z in UR's base frame, wood
    # colored like the real one so the camera has to tell it from the beam
    if args.get('table_rgba') is not None:
        pos, quat   = matrix_to_pos_quat(UR_BASE_IN_MJ_BASE @ pose_to_matrix([0, 0.6, args['table_z'] - 0.025, 0, 0, 0]))
        spec.body('base').add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[1.0, 1.0, 0.025], pos=pos, quat=quat,
                                   rgba=args['table_rgba'], contype=0, conaffinity=0)

    # the ChArUco board (hand_eye.py only) at board_pose (board frame in UR's base frame): its image stretched
    # over a plane. MuJoCo draws a plane from its +z side, so the plane is the board frame turned 180 deg about x
    if args.get('board_pose') is not None:
        px              = 100                           # texture pixels per square
        image           = board_image(make_board(args), args, px, args['board_margin'])
        texture         = spec.add_texture(name='board', type=mujoco.mjtTexture.mjTEXTURE_2D,
                                           width=image.shape[1], height=image.shape[0], nchannel=3)
        texture.data    = np.repeat(image[:, :, None], 3, axis=2).tobytes()
        material        = spec.add_material(name='board', texrepeat=[1, 1], texuniform=False)
        material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = 'board'
        half            = np.array(image.shape[::-1]) * args['square_len'] / px / 2     # plane half sizes (m)
        margin          = (half * 2 - np.array(args['board_squares']) * args['square_len']) / 2
        T_plane         = pose_to_matrix(args['board_pose']) @ pose_to_matrix([*(half - margin), 0, np.pi, 0, 0])
        pos, quat       = matrix_to_pos_quat(UR_BASE_IN_MJ_BASE @ T_plane)
        spec.body('base').add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[*half, 1], pos=pos, quat=quat,
                                   material='board', contype=0, conaffinity=0)
    return spec.compile()


def run_viewer(args):
    # runs in its own process, so its OpenGL window can not clash with the camera renderer and never slows
    # the servo loop: a mujoco-python-viewer window that follows the arm's joint angles until it is closed
    import mujoco_viewer
    signal.signal(signal.SIGINT, signal.SIG_IGN)    # Ctrl+C is for the main program, which ends this process
    receive = RTDEReceiveInterface(args['robot_ip'])
    model   = build_scene(args)
    data    = mujoco.MjData(model)
    viewer  = mujoco_viewer.MujocoViewer(model, data, title="URSim arm", width=1000, height=750)
    viewer.cam.lookat[:] = args['cam_lookat']
    viewer.cam.azimuth, viewer.cam.elevation, viewer.cam.distance = args['cam_azi'], args['cam_ele'], args['cam_dist']

    while viewer.is_alive:
        data.qpos[:] = receive.getActualQ()
        mujoco.mj_forward(model, data)
        viewer.render()         # one frame per call, paced by the screen refresh
    receive.disconnect()


class SimEnv:

    def __init__(self, args):
        # URSim is the robot, we only read its joint angles
        self.receive    = RTDEReceiveInterface(args['robot_ip'])

        # MuJoCo only draws the scene, nothing is ever stepped
        self.model      = build_scene(args)
        self.data       = mujoco.MjData(self.model)
        self.renderer   = mujoco.Renderer(self.model, args['height'], args['width'])

        # pinhole intrinsics of the MuJoCo camera, with the same fields as the RealSense ones
        f               = args['height'] / 2 / np.tan(np.radians(args['sim_fovy']) / 2)
        self.intrinsics = SimpleNamespace(fx=f, fy=f, ppx=(args['width'] - 1) / 2, ppy=(args['height'] - 1) / 2)

        # the beam box in UR's base frame and its half sizes, for contact
        self.T_beam     = pose_to_matrix(args['beam_pose'])
        self.beam_half  = np.array(args['beam_size']) / 2

        # optional viewer of the whole scene, a fresh (spawned) process that ends with this program
        if args['render']:
            multiprocessing.get_context('spawn').Process(target=run_viewer, args=(args,), daemon=True).start()

    def sync(self):
        # move the MuJoCo arm to URSim's joint angles and update every pose (kinematics only)
        self.data.qpos[:] = self.receive.getActualQ()
        mujoco.mj_forward(self.model, self.data)

    def render(self):
        # color (BGR, like the RealSense stream) and depth (m along the optical axis) from the wrist camera
        self.renderer.update_scene(self.data, camera='wrist_camera')
        color = np.ascontiguousarray(self.renderer.render()[:, :, ::-1])
        self.renderer.enable_depth_rendering()
        depth = self.renderer.render()
        self.renderer.disable_depth_rendering()
        return color, depth

    def first_contact(self, centers, radius : float, direction):
        # the balls (Nx3 centers, UR base frame) moving along direction: the travel (m) until the first one
        # touches a face of the beam box, and that face's plane (a point on it, outward normal), None if all miss
        c       = (centers - self.T_beam[:3, 3]) @ self.T_beam[:3, :3]      # centers in the box frame
        d       = self.T_beam[:3, :3].T @ direction
        best    = None
        for axis in range(3):
            for sign in (1, -1):
                if sign * d[axis] >= 0:
                    continue    # moving away from this face
                # travel until each ball reaches the face plane, kept only where it lands inside the face
                s       = (sign * c[:, axis] - self.beam_half[axis] - radius) / (-sign * d[axis])
                hit     = c + s[:, None] * d
                inside  = np.all(np.delete(np.abs(hit) <= self.beam_half, axis, axis=1), axis=1) & (s >= 0)
                if inside.any() and (best is None or s[inside].min() < best[0]):
                    normal  = sign * self.T_beam[:3, axis]
                    best    = (s[inside].min(), self.T_beam[:3, 3] + self.beam_half[axis] * normal, normal)
        return best

    def close(self):
        self.renderer.close()
        self.receive.disconnect()
