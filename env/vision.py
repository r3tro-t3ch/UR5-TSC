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
# One camera interface for real and simulated runs. With sim set, frames are
# rendered by MuJoCo with the arm following URSim; otherwise they come from the
# RealSense D435. Either way: BGR color, depth in meters, pinhole intrinsics.

from env.sim_env import SimEnv
from test_vision import start_camera, get_frames


class Vision:

    def __init__(self, args):
        self.sim = args['sim']

        # simulated camera (MuJoCo + URSim) or the real one on the wrist
        if self.sim:
            self.sim_env    = SimEnv(args)
            self.intrinsics = self.sim_env.intrinsics
        else:
            self.pipeline, self.align, self.depth_scale, self.intrinsics = start_camera(args)

    def get_frames(self):
        # color (BGR) and depth (m, 0 = no reading) images, aligned pixel to pixel
        if self.sim:
            self.sim_env.sync()     # the rendered arm first moves to where URSim's arm is now
            return self.sim_env.render()
        return get_frames(self.pipeline, self.align, self.depth_scale)

    def close(self):
        if self.sim:
            self.sim_env.close()
        else:
            self.pipeline.stop()
