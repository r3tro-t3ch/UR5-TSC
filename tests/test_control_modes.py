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
# Verify both robot models against the expected actuator force laws, torque
# saturation, and inactive-group isolation. Check mode-switch targets, invalid
# commands, headless stepping, and joint-sized QP constraints.

"""Check actual MuJoCo forces, mode isolation, and joint-sized QPs."""

from pathlib import Path
import unittest
import mujoco as mj
import numpy as np
from env.actuator_modes import ActuatorModes
from env.ur_env import UR10eEnv


ROOT = Path(__file__).resolve().parents[1]


class ControlModesTest(unittest.TestCase):
    def load(self, robot):
        model = mj.MjModel.from_xml_path(str(ROOT / "env" / f"{robot}.xml"))
        data = mj.MjData(model)
        mj.mj_resetDataKeyframe(model, data, model.keyframe("home").id)
        return model, data, ActuatorModes(model, data)

    def test_actual_force_laws_and_inactive_groups(self):
        for robot in ("ur10e", "ur5e"):
            model, data, controls = self.load(robot)
            self.assertEqual((model.nv, model.nu), (6, 18))
            data.qvel[:] = np.linspace(-0.03, 0.03, 6)
            for mode in controls.GROUPS:
                with self.subTest(robot=robot, mode=mode):
                    controls.set_mode(mode)
                    ids = controls.ids[mode]
                    command = (np.arange(1., 7.) if mode == "torque" else
                               data.qpos + 0.002 if mode == "position" else
                               data.qvel + 0.01)
                    controls.set_command(command)
                    # Disabled servos must produce no forces, even with stale targets.
                    inactive = np.setdiff1d(np.arange(model.nu), ids)
                    data.ctrl[inactive] = 10
                    mj.mj_forward(model, data)
                    if mode == "torque":
                        expected = command
                    elif mode == "position":
                        expected = (model.actuator_gainprm[ids, 0] * (command - data.qpos)
                                    + model.actuator_biasprm[ids, 2] * data.qvel)
                    else:
                        expected = model.actuator_gainprm[ids, 0] * (command - data.qvel)
                    limits = model.actuator_forcerange[ids]
                    expected = np.clip(expected, limits[:, 0], limits[:, 1])
                    np.testing.assert_allclose(data.qfrc_actuator, expected, atol=1e-9)
                    np.testing.assert_allclose(data.actuator_force[inactive], 0)

    def test_limits_switch_targets_and_bad_commands(self):
        for robot in ("ur10e", "ur5e"):
            model, data, controls = self.load(robot)
            controls.set_command(np.full(6, 1e4))
            mj.mj_forward(model, data)
            np.testing.assert_allclose(data.qfrc_actuator, controls.torque_limits[:, 1])
            controls.set_mode("position")
            np.testing.assert_allclose(data.ctrl[controls.ids["position"]], data.qpos)
            data.qvel[:] = 0.2
            controls.set_mode("velocity")
            np.testing.assert_allclose(data.ctrl[controls.ids["velocity"]], data.qvel)
            for command in (None, np.zeros(18), np.full(6, np.nan)):
                with self.assertRaises(ValueError):
                    controls.set_command(command)
            with self.assertRaises(ValueError):
                controls.set_mode("unknown")
            self.assertEqual(controls.mode, "velocity")
            with self.assertRaises(RuntimeError):
                controls.require_torque_mode()

    def test_headless_environment_and_qp_dimensions(self):
        from controller.tsc_consistent import ConsistentTaskSpaceController
        from controller.tsc_inconsistent import InconsistentTaskSpaceController
        from controller.tsc_clf import CLFTaskSpaceController
        for robot in ("ur10e", "ur5e"):
            env = UR10eEnv({"xml_file": f"{robot}.xml", "is_render": False,
                           "use_pinnochio_dynamics": False, "cbf": False})
            try:
                self.assertIsNone(env.viewer)
                controller = ConsistentTaskSpaceController(env)
                G, h = controller.get_ineq_constraint()
                self.assertEqual(G.shape, (12, 6))
                np.testing.assert_allclose(h[:6], env.actuators.torque_limits[:, 1])
                tau = controller.get_action(np.zeros(6))
                self.assertEqual(tau.shape, (6,))
                env.step(tau)
                other = InconsistentTaskSpaceController(env)
                A, b = other.get_eq_constraint()
                G, h = other.get_ineq_constraint()
                self.assertEqual(A.shape, (6, 12))
                self.assertEqual(G.shape, (12, 12))
                _, tau = other.get_action(np.zeros(6), np.eye(6))
                self.assertEqual(tau.shape, (6,))
                clf = CLFTaskSpaceController(env, 1.0, np.eye(6), np.eye(6))
                G, h = clf.get_ineq_constraint(150, env.ee_pos, env.ee_vel,
                                              np.zeros(3), env.ee_w, np.zeros(6))
                self.assertEqual(G.shape, (13, 6))
                np.testing.assert_allclose(h[:6], np.minimum(150, env.actuators.torque_limits[:, 1]))
                for mode in ("position", "velocity"):
                    env.set_control_mode(mode)
                    with self.assertRaises(RuntimeError):
                        controller.get_action(np.zeros(6))
                    env.step()
                self.assertTrue(np.all(np.isfinite(env.data.qpos)))
            finally:
                env.stop()


if __name__ == "__main__":
    unittest.main()
