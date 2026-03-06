import pinocchio as pin
import numpy as np
import os

class UR5EnvPinocchio:

    EE_SITE = "attachment_site"

    def __init__(self, args):
        
        xml_directory   = os.getcwd() + "/env"
        self.xml_file   = args['xml_file']
        self.xml_path   = os.path.join(xml_directory,self.xml_file)

        self.model      = pin.buildModelFromMJCF(self.xml_path)
        self.data       = self.model.createData()

        self._q  = pin.neutral(self.model)
        self._v  = np.zeros(self.model.nv)

        self.ee_site_frame_id   = self.model.getFrameId(self.EE_SITE)

        self.set_state(self._q, self._v)

    def _fk(self, q: np.ndarray, v: np.ndarray, a: np.ndarray | None = None):
        if a is None:
            a = np.zeros(self.model.nv)
        pin.forwardKinematics(self.model, self.data, q, v, a)
        pin.updateFramePlacements(self.model, self.data)

    def M(self, q : np.ndarray):
        # Mass matrix
        pin.crba(self.model, self.data, q)
        M = np.array(self.data.M.copy())
        M = np.tril(M.T) + np.triu(M, 1)
        return M
    
    def C(self, q : np.ndarray, qdot : np.ndarray):
        # Coriolis, Centrifugal and gravity term
        # pin.
        return pin.rnea(self.model, self.data, q, qdot, np.zeros(self.model.nv))

    def J(self, q : np.ndarray) -> np.ndarray:
        # Compute jacobian
        pin.computeJointJacobians(self.model, self.data, q)
        J = pin.getFrameJacobian(
            self.model, self.data,
            self.ee_site_frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        return np.array(J)

    def set_state(self, q : np.ndarray, qdot : np.ndarray):
        self._fk(q, qdot)
        