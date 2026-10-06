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
# wrench), the controller's inverse kinematics and motion commands (TCP, moveJ,
# moveL, move until contact, force mode press, freedrive, speedL, stop, watchdog). The same
# RTDE connection drives the real UR arm and URSim, only the IP differs.

import time
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

    def ik(self, pose, q_near):
        # joint angles (rad) for a TCP pose, the solution closest to q_near, or None when it is out of reach
        if not self.control.getInverseKinematicsHasSolution(list(pose), list(q_near)):
            return None
        return np.array(self.control.getInverseKinematics(list(pose), list(q_near)))

    # commands

    def set_tcp(self, pose):
        # TCP offset from the tool flange [x, y, z, rx, ry, rz], used by every pose and speed below
        self.control.setTcp(list(pose))

    def move_j(self, q, speed : float, acc : float):
        # blocking joint space move (rad/s, rad/s^2); ur_rtde only returns False when it fails, so raise
        if not self.control.moveJ(list(q), speed, acc):
            raise RuntimeError(f"moveJ to {np.round(q, 3)} failed")

    def move_l(self, pose, speed : float, acc : float):
        # blocking straight line TCP move to pose [x, y, z, rx, ry, rz] (m/s, m/s^2), raises when it fails
        if not self.control.moveL(list(pose), speed, acc):
            raise RuntimeError(f"moveL to {np.round(pose, 3)} failed")

    def move_until_contact(self, pose, speed : float, acc : float):
        # straight line TCP move towards pose that stops at the first contact (the robot's own contact
        # detection, in the direction of motion), so pose limits the travel when nothing is touched;
        # returns True on contact. Progress < 0 and changed means the asynchronous move has ended
        direction   = np.concatenate((np.asarray(pose[:3]) - self.get_tcp_pose()[:3], np.zeros(3)))
        last        = self.control.getAsyncOperationProgress()
        self.control.startContactDetection(list(direction))
        self.control.moveL(list(pose), speed, acc, True)
        while not self.control.readContactDetection():
            progress = self.control.getAsyncOperationProgress()
            if progress < 0 and progress != last:
                break
            time.sleep(0.002)
        contact = self.control.stopContactDetection()
        self.control.stopL(acc)
        return contact

    def free_drive(self, on : bool):
        # freedrive: the arm can be moved by hand (like the pendant's freedrive button) until switched off
        if on:
            self.control.teachMode()
        else:
            self.control.endTeachMode()

    def zero_ft(self):
        # zero the wrist force/torque sensor, with the arm at rest and touching nothing
        self.control.zeroFtSensor()

    def press(self, wrench, duration : float, damping : float, free, limits):
        # force mode about the current TCP for duration (s): apply wrench ([Fx, Fy, Fz, Mx, My, Mz] in the TCP
        # frame, N and Nm) on the compliant axes set in free (1 = compliant), stiff in the rest (limits: speeds on
        # the compliant axes, allowed deviations on the others); returns the TCP poses and wrenches of the second
        # half, when the arm should be at rest
        frame   = list(self.get_tcp_pose())
        command = [float(w) for w in wrench]
        if len(command) != 6:
            raise ValueError(f"press needs a 6 value wrench, got {wrench}")
        poses   = []
        read    = []
        start   = time.time()
        self.control.forceModeSetDamping(damping)
        while time.time() - start < duration:
            t = self.control.initPeriod()
            self.control.forceMode(frame, [int(f) for f in free], command, 2, [float(l) for l in limits])
            if time.time() - start > duration / 2:
                poses.append(self.get_tcp_pose())
                read.append(self.get_tcp_force())
            self.control.waitPeriod(t)
        self.control.forceModeStop()
        return np.array(poses), np.array(read)

    def turn_until(self, axis, speed : float, acc : float, stop, timeout : float):
        # turn the TCP about axis (unit vector in the base frame, through the TCP) at speed (rad/s), position
        # controlled, until stop(TCP pose, TCP wrench) says so or timeout (s); returns whether stop did
        start   = time.time()
        stopped = False
        while time.time() - start < timeout:
            t = self.control.initPeriod()
            self.control.speedL([0, 0, 0] + [float(a * speed) for a in axis], acc, 0.1)
            if stop(self.get_tcp_pose(), self.get_tcp_force()):
                stopped = True
                break
            self.control.waitPeriod(t)
        self.control.speedStop(acc)
        return stopped

    def mean_tcp_force(self, n : int):
        # wrench at the TCP averaged over n control periods
        read = []
        for _ in range(n):
            t = self.control.initPeriod()
            read.append(self.get_tcp_force())
            self.control.waitPeriod(t)
        return np.mean(read, axis=0)

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
