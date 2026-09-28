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
# Demonstrate torque, position, or velocity actuation on the UR10e or UR5e
# with a small joint-space reference motion. Support rendered and headless
# runs; this demo does not apply a CBF or CLF safety filter.

"""Small joint-space demo for each actuator mode (no CBF or CLF)."""

import argparse
import numpy as np
from env.ur_env import UR10eEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=("ur10e", "ur5e"), default="ur10e")
    parser.add_argument("--mode", choices=("torque", "position", "velocity"), default="torque")
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--headless", action="store_true")
    cli = parser.parse_args()
    if not np.isfinite(cli.duration) or cli.duration <= 0:
        parser.error("--duration must be finite and positive")
    env = UR10eEnv({
        "xml_file": f"{cli.robot}.xml", "control_mode": cli.mode,
        "is_render": not cli.headless, "use_pinnochio_dynamics": False, "cbf": False,
    })
    home = env.data.qpos.copy()
    try:
        while env.is_alive and env.data.time < cli.duration:
            # A small, slow shoulder oscillation around the home configuration.
            t = env.data.time
            q_ref, v_ref = home.copy(), np.zeros(env.n_joints)
            q_ref[0] += 0.1 * np.sin(t)
            v_ref[0] = 0.1 * np.cos(t)
            if cli.mode == "position":
                command = q_ref
            elif cli.mode == "velocity":
                command = v_ref + 2.0 * (q_ref - env.data.qpos)
            else:
                command = (env.data.qfrc_bias - env.data.qfrc_passive
                           + 100.0 * (q_ref - env.data.qpos)
                           + 20.0 * (v_ref - env.data.qvel))
            env.step(command)
        print(f"{cli.robot}: {cli.mode} mode, simulated {env.data.time:.3f} s")
        print("Final joint positions:", np.round(env.data.qpos, 4))
    finally:
        env.stop()


if __name__ == "__main__":
    main()
