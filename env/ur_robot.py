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
# All ur_rtde calls in one place: sensor readings (joint angles, TCP pose and
# wrench) and motion commands (TCP, moveJ, speedL, stop, watchdog). The same
# RTDE connection drives the real UR arm and URSim, only the IP differs.

import numpy as np
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface


class URRobot:

    def __init__(self, args):
        # control uploads a script to the robot (needs Remote Control mode), receive only reads
        self.control = RTDEControlInterface(args['robot_ip'])
        self.receive = RTDEReceiveInterface(args['robot_ip'])

    # sensor readings

    def get_q(self):
        # joint angles (rad), base to wrist 3
        return np.array(self.receive.getActualQ())

    def get_tcp_pose(self):
        # TCP pose [x, y, z, rx, ry, rz] in the base frame (m, rotation vector)
        return np.array(self.receive.getActualTCPPose())

    def get_tcp_force(self):
        # wrench at the TCP [Fx, Fy, Fz, Tx, Ty, Tz] (N, Nm), always ~zero in URSim
        return np.array(self.receive.getActualTCPForce())

    # commands

    def set_tcp(self, pose):
        # TCP offset from the tool flange [x, y, z, rx, ry, rz], used by every pose and speed below
        self.control.setTcp(list(pose))

    def move_j(self, q, speed : float, acc : float):
        # blocking joint space move (rad/s, rad/s^2)
        self.control.moveJ(list(q), speed, acc)

    def speed_l(self, xd, acc : float, time : float):
        # TCP velocity [vx, vy, vz, wx, wy, wz] in the base frame (m/s, rad/s), returns at once;
        # time (s) must be longer than the loop period, with time = 0 PolyScope 5.26 stops after the ramp
        self.control.speedL(list(xd), acc, time)

    def stop(self, acc : float):
        # decelerate to standstill (m/s^2), blocks until stopped (~speed / acc), so not inside a watchdog loop
        self.control.speedStop(acc)

    def start_watchdog(self, hz : float):
        # the robot stops itself if kick_watchdog() is not called at least this often
        self.control.setWatchdog(hz)

    def kick_watchdog(self):
        self.control.kickWatchdog()

    def close(self):
        # stop the arm, end the uploaded control script and disconnect
        self.control.speedStop()
        self.control.stopScript()
        self.receive.disconnect()
