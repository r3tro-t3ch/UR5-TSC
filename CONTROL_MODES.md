# MuJoCo joint control modes

Both `env/ur10e.xml` and `env/ur5e.xml` provide torque, position, and velocity
actuators. Exactly one actuator group is enabled at a time. Torque is the default.
Requires the project's MuJoCo 3.5.0 environment (also tested with 3.4.0).

## Try each mode

From the repository root, using the Python environment containing the project dependencies:

```sh
python main_modes.py --robot ur10e --mode torque
python main_modes.py --robot ur10e --mode position
python main_modes.py --robot ur10e --mode velocity
python main_modes.py --robot ur5e --mode position --headless --duration 5
```

These are small joint-space demonstrations, not safety-filtered experiments.
The torque demo uses joint PD feedback with MuJoCo bias/passive-force compensation.
The velocity demo uses velocity feedforward plus position feedback.

## Environment API

```python
args['control_mode'] = 'torque'  # default
env = UR10eEnv(args)
env.step(tau)                   # six torques in N m

env.set_control_mode('position')
env.step(q_desired)             # six joint positions in radians

env.set_control_mode('velocity')
env.step(qdot_desired)          # six joint velocities in rad/s
```

Commands always follow shoulder pan, shoulder lift, elbow, wrist 1, wrist 2,
wrist 3 order. `env.step()` without a command retains the active command.
Commands are clipped to the active actuator command limits. Invalid shapes and
nonfinite values raise an error.

On switching, position targets initialize to current joint positions, velocity
targets to current velocities (within command limits), and torque targets to
zero. This clears stale commands but does not ensure continuous torque or a safe
transition. Supply the intended command before the next step. For comparisons,
choose a mode at reset rather than switching during a safety-critical motion.

## Actuator configuration

| Mode | Group | Command | Force law before saturation, with unit gear |
| --- | --- | --- | --- |
| Torque | 0 | N m | `tau = u` |
| Position | 1 | rad | `tau = kp*(u-q) - kd*qdot` |
| Velocity | 2 | rad/s | `tau = kv*(u-qdot)` |

The XML uses explicit settings on each actuator, without servo-class inheritance.
Position gains preserve the old model values. Velocity gains use the old damping
gains; their command limits are illustrative +/-1 rad/s and should be tuned for
the experiment. Force limits preserve the existing model values:

- UR10e: `[330, 330, 150, 56, 56, 56]` N m.
- UR5e: `[150, 150, 150, 28, 28, 28]` N m.

These are simulation settings, not independently validated hardware limits.
Joint damping, armature, collision geometry and joint limits are unchanged.

The models have `model.nu == 18` actuator inputs and `model.nv == 6` joint
velocities. Controllers use `env.n_joints` for six-dimensional joint commands.
`env.actuators.ids[mode]` maps them into `data.ctrl`; do not assign a six-vector
directly to the 18-element `data.ctrl`. Home keyframes include all 18 controls.

## Existing QP experiments

`main.py`, `main_clf.py`, and `main_contraction.py` explicitly select torque mode.
Their torque QPs retain six torque variables and use the physical actuator
limits. A supplied `tau_max` further tightens those limits, including the CLF
example's 150 N m cap. Torque controllers reject position/velocity modes and
raise an error if the solver returns no solution; that error is not a certified
backup controller.

Switching actuator modes does not reformulate a torque CBF into a position or
velocity CBF. This change does not correct or validate the pre-existing CBF/CLF
derivations, Pinocchio/MuJoCo dynamics agreement, contraction claims, or tracking
gains. In particular, the previously identified Jacobian-rate and CLF dynamics
issues remain separate work. Expect tracking behavior to change now that QP
torques are applied directly.

## Verification

```sh
python -m unittest discover -s tests -v
```

Tests compare MuJoCo's actual actuator torques against each force law on both
robots, check saturation and inactive-group isolation, exercise runtime
switching, and verify the existing QP dimensions and torque limits headlessly.
QP tests require `qpsolvers` and `cvxopt`. Headless mode demos do not require
Pinocchio; the existing experiments with `use_pinnochio_dynamics=True` do.
