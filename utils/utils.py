import numpy as np
from scipy.spatial.transform import Rotation as R

def quat2euler(quat):
    _quat = np.concatenate([quat[1:], quat[:1]])
    r = R.from_quat(_quat)
    euler = r.as_euler('xyz', degrees=False)
    return euler

def skew_symmetric(vector):
    mat = np.zeros((vector.shape[0], vector.shape[0]))

    mat[0,1] = -vector[2]
    mat[0,2] = vector[1]
    mat[1,0] = vector[2]
    mat[1,2] = -vector[0]
    mat[2,0] = -vector[1]
    mat[2,1] = vector[0]

    return mat
