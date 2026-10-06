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
# The ChArUco calibration board used by hand_eye.py: the board from args, an
# image of it (to print, or as the simulated board's texture) and its corners
# found in a color image. Board frame (OpenCV): origin at the top-left corner of
# the squares as printed, x to the right, y down, z into the board.

import cv2
import numpy as np


def make_board(args):
    # board_squares (columns, rows) squares of square_len (m), an ArUco marker of marker_len in every white one
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, args['board_dict']))
    return cv2.aruco.CharucoBoard(args['board_squares'], args['square_len'], args['marker_len'], dictionary)


def board_image(board, args, px_per_square : int, margin : float):
    # grayscale image of the board, px_per_square pixels per square and a white margin (m) all round
    m       = round(margin / args['square_len'] * px_per_square)
    cols, rows = args['board_squares']
    return board.generateImage((cols * px_per_square + 2 * m, rows * px_per_square + 2 * m), marginSize=m)


def detect_board(detector, board, color : np.ndarray, min_corners : int):
    # chessboard corners seen in a color image: pixels (Nx2), ids and points in the board frame (Nx3),
    # None when fewer than min_corners are found
    corners, ids, _, _ = detector.detectBoard(color)
    if ids is None or len(ids) < min_corners:
        return None
    ids = ids.ravel()
    return {'uv': corners.reshape(-1, 2), 'ids': ids, 'obj': board.getChessboardCorners()[ids]}
