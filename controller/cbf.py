import numpy as np

class CBF:

    def __init__(self, obstacle : np.ndarray = None, alpha : np.ndarray = None, obstacle_r = None):

        self.obstacle   = obstacle
        self.alpha      = alpha
        self.obstacle_r = obstacle_r

    def h(self, x):
        return (x - self.obstacle).T @ (x - self.obstacle) - self.obstacle_r**2
    
    def h_x_q(self, x : np.ndarray, J : np.ndarray):
        return 2*(x - self.obstacle) @ J

    def h_dot_q(self, x : np.ndarray, q_dot : np.ndarray, J : np.ndarray):
        return 2 * (x - self.obstacle).T @ J @ q_dot + self.alpha[0] * self.h(x)

    def get_cbf_ineq_constraints_q(self, x : np.ndarray, q_dot : np.ndarray, J : np.ndarray, M_inv : np.ndarray, C : np.ndarray):
        C_cbf = -2 * (x - self.obstacle).T @ J @ M_inv

        c_cbf =     self.alpha[1] * self.h_dot_q(x, q_dot, J) \
                +   2 * q_dot.T @ J.T @ J @ q_dot \
                +   self.alpha[0] * self.h_x_q(x, J) @ q_dot \
                -   self.h_x_q(x, J) @ M_inv @ C
        

        return C_cbf[np.newaxis, :], np.array([c_cbf])