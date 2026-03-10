import pinocchio as pin
import pinocchio.casadi as cpin
import casadi as ca
import numpy as np
import os

class UR5EnvPinocchio:

    EE_SITE         = "attachment_site"
    WRIST_3_LINK    = "wrist_3_link"

    def __init__(self, args):
        
        xml_directory   = os.getcwd() + "/env"
        self.xml_file   = args['xml_file']
        self.xml_path   = os.path.join(xml_directory,self.xml_file)

        # Pinocchio data types
        self.model      = pin.buildModelFromMJCF(self.xml_path)
        self.data       = self.model.createData()

        # Pinocchio CaSaDi data types
        self.cmodel     = cpin.Model(self.model)
        self.cdata      = self.cmodel.createData()

        # CaSaDi symbolic variables
        self._setup_symbolic_derivatives()

        # init states and numerical values
        self.q          = pin.neutral(self.model)
        self.q_dot      = np.zeros(self.model.nv)

        # frames for position and orientation of end effector
        self.ee_site_frame_id       = self.model.getFrameId(self.EE_SITE)
        self.wrist_3_link_frame_id  = self.model.getFrameId(self.WRIST_3_LINK)

        self.set_state(self.q, self.q_dot)

    # Mass matrix and it's derivatives
    def M(self, q : np.ndarray):
        # Mass matrix
        pin.crba(self.model, self.data, q)
        M = np.array(self.data.M.copy())
        M = np.tril(M.T) + np.triu(M, 1)
        return M
    
    def Minv(self, q : np.ndarray):
        # Mass matrix
        pin.crba(self.model, self.data, q)
        M = np.array(self.data.M.copy())
        M = np.tril(M.T) + np.triu(M, 1)
        return np.linalg.inv(M)
    
    def dMdq(self, q):
        # computes \frac{\del M}{\del q}
        nq = self.model.nq
        return np.array(self._dMdq_fn_c(q)).reshape((nq, nq, nq))
    
    def dMinvdq(self, q):
        # computes \frac{\del M_inv}{\del q}
        nq = self.model.nq
        return np.array(self._dMinvdq_fn_c(q)).reshape((nq, nq, nq))
    
    # Task space inertia
    def Lambda(self, q : np.ndarray):
        J       = self.J(q)
        M_inv   = self.Minv(q)

        Lambda_inv  = J @ M_inv @ J.T
        Lambda      = np.linalg.inv(Lambda_inv + np.identity(Lambda_inv.shape[0]) * 1e-5)

        return Lambda

    # Coriolis, Centrifugal and gravity term and their derivatives
    def C(self, q : np.ndarray, qdot : np.ndarray):
        # Coriolis, Centrifugal and gravity term
        return np.array(pin.rnea(self.model, self.data, q, qdot, np.zeros(self.model.nv)))

    def dCdq(self, q : np.ndarray, q_dot : np.ndarray):
        # copmutes \frac{del C}{\del q}
        return np.array(self._dCdq_fn_c(q, q_dot))
    
    def dCdqdot(self, q : np.ndarray, q_dot : np.ndarray):
        # copmutes \frac{del C}{\del q_dot}
        return np.array(self._dCdqdot_fn_c(q, q_dot))
    
    def J(self, q : np.ndarray) -> np.ndarray:
        # Compute jacobian
        pin.computeJointJacobians(self.model, self.data, q)
        J = pin.getFrameJacobian(
            self.model, self.data,
            self.ee_site_frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        return np.array(J)
    
    def Jdot(self, q : np.ndarray, qdot : np.ndarray):
        # copmute J dot
        pin.computeJointJacobiansTimeVariation(self.model, self.data, q, qdot)
        Jdot = pin.getFrameJacobianTimeVariation(
            self.model,
            self.data,
            self.ee_site_frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        return np.array(Jdot)

    def get_ee_pose(self, mujoco=True):

        T = self.data.oMf[self.ee_site_frame_id]  # SE3 transform
        R = self.data.oMf[self.wrist_3_link_frame_id]
    
        pos  = T.translation                       # (3,)  xyz
        quat = pin.SE3ToXYZQUAT(R)[3:]             # (4,)  xyzw
        if mujoco:
            quat = np.roll(quat, 1)
            # w is negated 
            if quat[0] < 0:
                quat = -quat

        return pos, quat

    def set_state(self, q : np.ndarray, qdot : np.ndarray):
        self._fk(q, qdot)
        
    def _setup_symbolic_derivatives(self):
        
        self.q_sx       = ca.SX.sym("q", self.model.nq)
        self.q_dot_sx   = ca.SX.sym("q_dot", self.model.nv)

        q, q_dot = self.q_sx, self.q_dot_sx
        nq, nv = self.model.nq, self.model.nv

        # M 
        cpin.crba(self.cmodel, self.cdata, q)
        M_sx    = self.cdata.M

        M_sx        = (M_sx.T + M_sx)/2
        M_sx_inv    = ca.inv(M_sx)

        dMdq_sx     = ca.jacobian(M_sx, q)
        dMinvdq_sx  = ca.jacobian(M_sx_inv, q)

        # C
        cpin.rnea(self.cmodel, self.cdata, q, q_dot, ca.SX.zeros(nv))
        C_sx    = self.cdata.tau

        dCdq    = ca.jacobian(C_sx, q)
        dCdqdot = ca.jacobian(C_sx, q_dot)

        # setup functions
        self._dMdq_fn_c     = ca.Function("dMdq",   [q],            [dMdq_sx])
        self._dMinvdq_fn_c  = ca.Function("dMinvdq",[q],            [dMinvdq_sx])
        self._dCdq_fn_c     = ca.Function("dCdq",   [q, q_dot],     [dCdq])
        self._dCdqdot_fn_c  = ca.Function("dCdqdot",[q, q_dot],     [dCdqdot])

    def _fk(self, q: np.ndarray, q_dot: np.ndarray, q_ddot: np.ndarray | None = None):
        if q_ddot is None:
            q_ddot = np.zeros(self.model.nv)
        pin.forwardKinematics(self.model, self.data, q, q_dot, q_ddot)
        pin.updateFramePlacements(self.model, self.data)