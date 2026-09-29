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
# Detect the flat faces of a wooden beam in front of an Intel RealSense D435.
# Back-project the depth image to a point cloud and fit planes to it one at a
# time with RANSAC. Measure each face (midpoint, its distance, length and
# breadth) and show it in a cv2 window and its point cloud in an Open3D window.
# Press q or Esc to quit.

import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs

# overlay colors (BGR), one per detected face, all readable on a white background and under black text
COLORS = [(0, 0, 255), (0, 170, 0), (255, 144, 30), (0, 165, 255), (255, 0, 255)]


def start_camera(args):
    # stream depth and color, and align the depth image to the color image
    pipeline    = rs.pipeline()
    config      = rs.config()
    config.enable_stream(rs.stream.depth, args['width'], args['height'], rs.format.z16, args['fps'])
    config.enable_stream(rs.stream.color, args['width'], args['height'], rs.format.bgr8, args['fps'])
    profile     = pipeline.start(config)

    # depth scale converts the raw depth units to meters
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    intrinsics  = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    return pipeline, rs.align(rs.stream.color), depth_scale, intrinsics


def get_frames(pipeline, align, depth_scale):
    frames  = align.process(pipeline.wait_for_frames())
    color   = np.asanyarray(frames.get_color_frame().get_data())
    depth   = np.asanyarray(frames.get_depth_frame().get_data()) * depth_scale  # meters, 0 = no reading
    return color, depth


def get_points(depth, intrinsics):
    # back-project every pixel (u, v, depth) to a 3D point (x, y, z) in the camera frame
    v, u    = np.indices(depth.shape)
    x       = (u - intrinsics.ppx) / intrinsics.fx * depth
    y       = (v - intrinsics.ppy) / intrinsics.fy * depth
    return np.dstack((x, y, depth))


def fit_plane(points, args, rng):
    # RANSAC: try planes through 3 random points, keep the one with the most inliers
    best_inliers = None
    for _ in range(args['ransac_iters']):
        p0, p1, p2 = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(p1 - p0, p2 - p0)
        if np.linalg.norm(normal) < 1e-9:
            continue    # the 3 points are collinear, no plane
        normal /= np.linalg.norm(normal)
        inliers = np.abs((points - p0) @ normal) < args['plane_tol']
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers

    # refine with a least squares fit: the normal is the direction of least variance
    centroid    = points[best_inliers].mean(axis=0)
    normal      = np.linalg.eigh(np.cov((points[best_inliers] - centroid).T))[1][:, 0]
    if normal @ centroid > 0:
        normal = -normal    # point the normal towards the camera
    return normal, -normal @ centroid


def largest_patch(mask):
    # remove speckle, then keep only the biggest connected region of the mask
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n_labels < 2:
        return np.zeros(mask.shape, dtype=bool)     # nothing but background
    return labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])


def measure_face(face_points):
    # in-plane axes are the two directions with the most spread, the third one is the plane normal
    center  = face_points.mean(axis=0)
    axes    = np.linalg.eigh(np.cov((face_points - center).T))[1][:, 1:]    # 3x2
    flat    = ((face_points - center) @ axes).astype(np.float32)            # Nx2 in-plane coordinates

    # smallest rectangle around the face: its sides are the length and breadth
    rect                = cv2.minAreaRect(flat)
    (u, v), (w, h), _   = rect
    midpoint            = center + np.array([u, v]) @ axes.T
    corners             = center + cv2.boxPoints(rect) @ axes.T             # 4x3

    return {'midpoint': midpoint, 'distance': np.linalg.norm(midpoint),
            'length': max(w, h), 'breadth': min(w, h), 'corners': corners}


def detect_faces(depth, points, args, rng):
    # only look at pixels with a valid depth reading inside the working range
    remaining   = (depth > args['depth_min']) & (depth < args['depth_max'])
    faces       = []

    for _ in range(args['max_faces']):
        candidates = points[remaining]
        if len(candidates) < args['min_pixels']:
            break

        # fit on a random subset to keep RANSAC fast, then find every pixel on that plane
        n_samples       = min(len(candidates), args['ransac_points'])
        subset          = candidates[rng.choice(len(candidates), n_samples, replace=False)]
        normal, offset  = fit_plane(subset, args, rng)
        on_plane        = remaining & (np.abs(points @ normal + offset) < args['plane_tol'])

        # one face is one connected patch of the plane
        mask = largest_patch(on_plane)
        if mask.sum() < args['min_pixels']:
            break

        faces.append({'mask': mask, 'points': points[mask], 'normal': normal, **measure_face(points[mask])})
        remaining &= ~mask  # look for the next face in what is left

    # nearest first, so a face keeps its number and color from frame to frame
    return sorted(faces, key=lambda face: face['distance'])


def draw_faces(color, faces, intrinsics):
    # tint each face, then outline it
    tinted = color.copy()
    for i, face in enumerate(faces):
        tinted[face['mask']] = COLORS[i % len(COLORS)]
    output = cv2.addWeighted(tinted, 0.5, color, 0.5, 0)

    for i, face in enumerate(faces):
        contours, _ = cv2.findContours(face['mask'].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(output, contours, -1, COLORS[i % len(COLORS)], 2)

        # mark the midpoint of the face (3D point -> pixel) and write its report next to it
        x, y, z = face['midpoint']
        u, v    = int(x / z * intrinsics.fx + intrinsics.ppx), int(y / z * intrinsics.fy + intrinsics.ppy)
        cv2.circle(output, (u, v), 5, (255, 255, 255), -1)

        report = [f"face {i}: {face['distance']:.2f} m",
                  f"xyz {x:.2f} {y:.2f} {z:.2f} m",
                  f"{face['length']:.2f} x {face['breadth']:.2f} m"]

        # keep the text block inside the image
        text_width      = max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] for line in report)
        text_u, text_v  = np.clip(u + 8, 4, output.shape[1] - text_width - 4), np.clip(v, 15, output.shape[0] - 45)
        for k, line in enumerate(report):
            cv2.putText(output, line, (int(text_u), int(text_v) + 20 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return output


def start_viewer():
    # Open3D window with a white background that we update ourselves every frame
    app     = o3d.visualization.gui.Application.instance
    app.initialize()
    window  = app.create_window("face point clouds", 800, 600)
    widget  = o3d.visualization.gui.SceneWidget()
    widget.scene = o3d.visualization.rendering.Open3DScene(window.renderer)
    widget.scene.set_background([1.0, 1.0, 1.0, 1.0])
    widget.scene.view.set_post_processing(False)    # no tone mapping, so white is pure white and colors are exact
    window.add_child(widget)

    # the 3D scene fills the whole window
    def on_layout(context):
        widget.frame = window.content_rect
    window.set_on_layout(on_layout)

    # start at the sensor looking along its z axis (y points down in the camera frame)
    widget.setup_camera(60.0, o3d.geometry.AxisAlignedBoundingBox([-2, -2, 0], [2, 2, 4]), [0, 0, 0.8])
    widget.scene.camera.look_at([0, 0, 0.8], [0, 0, 0], [0, -1, 0])
    return app, widget


def update_viewer(widget, faces, labels, args):
    # materials: unlit, so the colors stay exactly as painted
    point_material              = o3d.visualization.rendering.MaterialRecord()
    point_material.shader       = "defaultUnlit"
    point_material.point_size   = 4
    line_material               = o3d.visualization.rendering.MaterialRecord()
    line_material.shader        = "unlitLine"
    line_material.line_width    = 3

    # remove last frame's geometry (removing a name that does not exist is harmless) and labels
    for i in range(args['max_faces']):
        widget.scene.remove_geometry(f"face_{i}")
        widget.scene.remove_geometry(f"outline_{i}")
    for label in labels:
        widget.remove_3d_label(label)

    labels = []
    for i, face in enumerate(faces):
        # point cloud of the face in the same color as its tint in the cv2 window (BGR -> RGB)
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(face['points']))
        cloud.paint_uniform_color(np.array(COLORS[i % len(COLORS)][::-1]) / 255)

        # measured rectangle around the face, black so it shows on the white background
        outline = o3d.geometry.LineSet(o3d.utility.Vector3dVector(face['corners']),
                                       o3d.utility.Vector2iVector([[0, 1], [1, 2], [2, 3], [3, 0]]))
        outline.paint_uniform_color([0, 0, 0])
        widget.scene.add_geometry(f"face_{i}", cloud, point_material)
        widget.scene.add_geometry(f"outline_{i}", outline, line_material)

        # length and breadth at the midpoint, in black (the default label color is white)
        label       = widget.add_3d_label(face['midpoint'], f"{face['length']:.2f} x {face['breadth']:.2f} m")
        label.color = o3d.visualization.gui.Color(0, 0, 0)
        labels.append(label)

    return labels


def main(args):
    pipeline, align, depth_scale, intrinsics = start_camera(args)
    app, widget = start_viewer()
    rng     = np.random.default_rng(0)  # fixed seed, so the same frame always gives the same faces
    labels  = []                        # the 3D labels currently shown, removed again on the next frame

    try:
        while True:
            color, depth    = get_frames(pipeline, align, depth_scale)
            faces           = detect_faces(depth, get_points(depth, intrinsics), args, rng)
            cv2.imshow("beam faces", draw_faces(color, faces, intrinsics))
            labels          = update_viewer(widget, faces, labels, args)

            # q, Esc or closing the Open3D window quits
            if not app.run_one_tick() or cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

        # close the Open3D window, it needs one more tick to finish closing (else Python crashes on exit)
        app.quit()
        app.run_one_tick()


if __name__ == "__main__":

    args = {}

    # camera
    args['width']           = 640
    args['height']          = 480
    args['fps']             = 30

    # working range in front of the camera (m)
    args['depth_min']       = 0.3
    args['depth_max']       = 1.5

    # face detection
    args['max_faces']       = 5         # planes to look for, a beam shows up to 3 of them, the rest is clutter
    args['plane_tol']       = 0.01      # max distance of a point to its plane (m)
    args['ransac_iters']    = 200
    args['ransac_points']   = 2000      # points RANSAC samples from
    args['min_pixels']      = 3000      # smallest patch counted as a face

    main(args)
