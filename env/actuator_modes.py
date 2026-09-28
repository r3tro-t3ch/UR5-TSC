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
# Map joint-sized commands to mutually exclusive torque, position, and
# velocity actuator groups. Initialize targets on mode switches, validate and
# clip commands, and expose joint torque constraints for QP controllers.

"""Joint commands for the mutually exclusive MuJoCo actuator groups."""

import mujoco as mj
import numpy as np


class ActuatorModes:
    GROUPS = {"torque": 0, "position": 1, "velocity": 2}

    def __init__(self, model, data, mode="torque"):
        self.model, self.data = model, data
        self.ids = {
            mode: np.flatnonzero(model.actuator_group == group)
            for mode, group in self.GROUPS.items()
        }
        torque_ids = self.ids["torque"]
        self.n_joints = len(torque_ids)
        if self.n_joints != model.nv or model.nq != model.nv:
            raise ValueError("Expected a fixed-base, fully actuated hinge-joint arm")
        self.joint_ids = model.actuator_trnid[torque_ids, 0]
        self.qpos_ids = model.jnt_qposadr[self.joint_ids]
        self.dof_ids = model.jnt_dofadr[self.joint_ids]
        if not np.array_equal(self.dof_ids, np.arange(model.nv)):
            raise ValueError("Actuator order must match generalized velocity order")
        for ids in self.ids.values():
            if (len(ids) != self.n_joints
                    or not np.array_equal(model.actuator_trnid[ids, 0], self.joint_ids)):
                raise ValueError("Every mode must actuate the same joints in the same order")
        self.mode = None
        self.set_mode(mode)

    @property
    def command_limits(self):
        return self.model.actuator_ctrlrange[self.ids[self.mode]].copy()

    @property
    def torque_limits(self):
        ids = self.ids["torque"]
        ctrl = self.model.actuator_ctrlrange[ids]
        force = self.model.actuator_forcerange[ids]
        return np.column_stack((np.maximum(ctrl[:, 0], force[:, 0]),
                                np.minimum(ctrl[:, 1], force[:, 1])))

    def set_mode(self, mode):
        if mode not in self.GROUPS:
            raise ValueError(f"Unknown control mode {mode!r}; choose {tuple(self.GROUPS)}")
        # Initialize targets before enabling the destination group. This avoids
        # stale targets, but is not a guarantee of continuous torque at a switch.
        target = (self.data.qpos[self.qpos_ids].copy() if mode == "position"
                  else self.data.qvel[self.dof_ids].copy() if mode == "velocity"
                  else np.zeros(self.n_joints))
        self.data.ctrl[:] = 0
        self.mode = mode
        self.set_command(target)
        mask = sum(1 << group for group in self.GROUPS.values())
        disabled = mask & ~(1 << self.GROUPS[mode])
        self.model.opt.disableactuator = (self.model.opt.disableactuator & ~mask) | disabled
        mj.mj_forward(self.model, self.data)

    def set_command(self, command):
        command = np.asarray(command, dtype=float)
        if command.shape != (self.n_joints,) or not np.all(np.isfinite(command)):
            raise ValueError(f"Expected {self.n_joints} finite joint commands in {self.mode} mode")
        limits = self.command_limits
        self.data.ctrl[self.ids[self.mode]] = np.clip(command, limits[:, 0], limits[:, 1])

    def require_torque_mode(self):
        if self.mode != "torque":
            raise RuntimeError("This controller outputs torques; select torque mode before using it")

    def torque_constraints(self, tau_max=None):
        """G, h for G @ tau <= h, optionally with a tighter controller cap."""
        limits = self.torque_limits
        if tau_max is not None:
            cap = np.broadcast_to(np.asarray(tau_max, dtype=float), (self.n_joints,))
            if not np.all(np.isfinite(cap)) or np.any(cap <= 0):
                raise ValueError("tau_max must contain finite positive limits")
            limits[:, 0] = np.maximum(limits[:, 0], -cap)
            limits[:, 1] = np.minimum(limits[:, 1], cap)
        eye = np.eye(self.n_joints)
        return np.vstack((eye, -eye)), np.concatenate((limits[:, 1], -limits[:, 0]))
