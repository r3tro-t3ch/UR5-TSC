import numpy as np
from env.ur5_pinocchio_env import UR5EnvPinocchio

class Contraction:

    def __init__(self, pin_env : UR5EnvPinocchio):
        
        # Pinnochio environment
        self.pin_env    = pin_env

    def dfdx(self, q : np.ndarray, q_dot : np.ndarray):

        C       = self.pin_env.C(q, q_dot)
        Minv    = self.pin_env.Minv(q)

        # derivatives wrt q
        dMinvdq = self.pin_env.dMinvdq(q)
        dCdq    = self.pin_env.dCdq(q, q_dot)

        # derivatives wrt qdot
        dCdqdot = self.pin_env.dCdqdot(q, q_dot)

        # derivatives wrt both states
        # here the tensor operation goes as follows
        # dM_inv/dq_i @ C -> (ixj) @ (jx1) = (ix1) gives basis vectors
        # so now we have k number of (ix1) basis vector
        # we can think of this multiplication as happening on k index and bought forward so it is (i x k)
        dfdq    = - np.einsum('ijk,j->ik', dMinvdq, C) - Minv @ dCdq
        dfdqdot = - Minv @ dCdqdot

        zeros   = np.zeros((self.pin_env.model.nq, self.pin_env.model.nq))
        I       = np.identity(self.pin_env.model.nq)

        dfdx = np.block([
            [zeros, I],
            [dfdq, dfdqdot]
        ])

        return dfdx
    
    def dgudx(self, q : np.ndarray, tau : np.ndarray):

        dMinvdq = self.pin_env.dMinvdq(q)
        zeros   = np.zeros((self.pin_env.model.nq, self.pin_env.model.nq))

        # dM_inv/dq @ tau
        # we can think of this tensor vector operation as follows
        # tau has k elements and dMdq has k (i x j) matrices
        # each scalar tau_k is multipled with kth dM_inv/dq and added together
        # this results in a matrix (i x j)
        dgudq   = np.einsum('ijk,k->ij', dMinvdq, tau)

        dgudx = np.block([
            [zeros, zeros],
            [dgudq, zeros]
        ])

        print("dgudx : ", dgudx.shape)

        return dgudx
    
    def A(self, q : np.ndarray, q_dot : np.ndarray, tau : np.ndarray):
        return self.dfdx(q, q_dot) + self.dgudx(q, tau)

    def W(self, q):
        return self.pin_env.Minv(q)
    
    def W_dot(self, q, q_dot):

        Minv    = self.pin_env.Minv(q)

        dMinvdq = self.pin_env.dMinvdq(q)
        print(dMinvdq.shape)

        # M(q)
        # Mdot = dMdq @ qdot
        # we can think of this tensor vector operation as follows
        # qdot has k elements and dMdq has k (i x j) matrices
        # each scalar qdot_k is multipled with kth dMdq and added together
        # this results in a matrix (i x j)
        M_dot    = np.einsum('ijk,k->ij', dMinvdq, q_dot) 

        return - Minv @ M_dot @ Minv
    
    def contraction_condition(self, q, q_dot, tau, _lambda):

        W       = np.kron(np.identity(2), self.W(q))
        W_dot   = np.kron(np.identity(2), self.W_dot(q, q_dot))
        A       = self.A(q, q_dot, tau)

        C       = W_dot - A @ W - W @ A.T + 2 * _lambda * W

        eigvals = np.linalg.eigvalsh(C)

        return C, eigvals, np.all(eigvals > 0)


    
