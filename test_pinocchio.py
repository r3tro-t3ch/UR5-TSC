import numpy as np
from env.ur_pinocchio_env import UR5EnvPinocchio
from env.ur_env import UR10eEnv
import mujoco


args = {}
args['is_render']   = True
args['xml_file']    = 'ur5e.xml'
args['cam_azi']     = 90
args['cam_ele']     = -20
args['cam_dist']    =  5

args['des_pos']     = np.array([0.6,0.6,0.6])
args['des_ori_q']   = np.array([1, 0.0, 0.0, 0.0])

# cbf
args['cbf']             = False

# pin_env     = UR5EnvPinocchio(args)
mj_env      = UR10eEnv(args)

q = np.ones(6) * 1.57
v = np.zeros(6)

# MuJoCo
mj_env.data.qpos[:] = q
mj_env.data.qvel[:] = v
# mujoco.mj_forward(mj_env.model, mj_env.data)
# mujoco.mj_crb(mj_env.model, mj_env.data)   # fill qM
mj_env.update_robot_states()               # recompute M, C, J, Lambda, mu

# # Pinocchio
# pin_env.set_state(q, v)

# M_pin = pin_env.M(q)

# M_mj = mj_env.M

# print("M : ", M_mj, M_pin)
# print("Difference:")
# print(M_pin - M_mj)

# print("Close?")
# print(np.allclose(M_pin, M_mj, atol=1e-5))

# print("C : ", mj_env.C, pin_env.C(q, v))
# print(np.allclose(mj_env.C, pin_env.C(q, v), atol=1e-5))

# J = np.concatenate([mj_env.jacp, mj_env.jacr])

# print(pin_env.J(q), J)
# print(np.allclose(pin_env.J(q), J, atol=1e-5))

