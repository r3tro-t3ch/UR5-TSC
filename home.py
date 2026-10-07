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
# Move the arm to the home pose that visual_servo.py starts from (its home_q),
# the same for the real arm and URSim. Refuses a home pose near a singularity
# or the table, and a joint move there that takes any part of the arm or tool
# near the table (env/clearance.py); otherwise one joint move at move_speed.

import numpy as np
from env.clearance import load_arm, joint_path, problems, table_problem
from env.ur_robot import URRobot
from utils.utils import ur_branch


def main(args):
    # home itself must be clear of the singularities and the table
    arm     = load_arm(args)
    home    = np.array(args['home_q'])
    near    = problems(home, ur_branch(home), arm, args['start_margins'], args)
    if near:
        raise SystemExit(f"home_q is not safe ({', '.join(near)}), update it")

    robot   = URRobot(args)
    try:
        # and so must the way there, checked every step_angle (a joint move: only the table matters on the way)
        for q in joint_path(robot.get_q(), home, args['step_angle']):
            if table_problem(q, arm, args):
                raise SystemExit(f"the move home takes the {table_problem(q, arm, args)[0]}, move the arm up "
                                 "(freedrive) first")
        robot.move_j(home, args['move_speed'], args['move_acc'])
        print("at home")
    finally:
        robot.close()


if __name__ == "__main__":

    args = {}

    # real arm, or URSim (docker, ports on localhost)
    args['sim']             = False
    args['robot_ip']        = "127.0.0.1" if args['sim'] else "192.168.1.100"

    # home (rad), keep it the same as visual_servo.py's home_q (wrist above the tool), and the same guard:
    # singularity margins (shoulder in m, elbow and wrist as |sin|) and the table (top at table_z in UR's base
    # frame, nothing closer than table_margin, shapes from xml_file), the move checked every step_angle (rad)
    args['home_q']          = [1.5823, -1.4434, -2.1595, 2.0374, -1.5729, -0.0185]
    args['start_margins']   = {'shoulder': 0.08, 'elbow': 0.17, 'wrist': 0.17}
    args['xml_file']        = 'ur10e.xml'
    args['table_z']         = 0.0
    args['table_margin']    = 0.05
    args['step_angle']      = np.radians(2)
    args['move_speed']      = 0.5       # rad/s
    args['move_acc']        = 0.5       # rad/s^2

    main(args)
