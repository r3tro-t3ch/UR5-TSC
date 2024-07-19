import numpy as np

class TrajectoryGenerator():
    def __init__(self,dt):
        self.t = 0
        self.dt = dt
        self._coeff_x = np.zeros((4,)) # 3rd order poly
        self._coeff_y = np.zeros((4,)) # 3rd order poly
        self._coeff_z = np.zeros((4,)) # 3rd order poly
        
    def reset_trajectory(self,start_pos,end_pos,start_vel,end_vel,T):
        self.start_pos = start_pos
        x0,y0,z0 = start_pos[0],start_pos[1],start_pos[2]
        vx0,vy0,vz0 = start_vel[0],start_vel[1],start_vel[2]
        xT,yT,zT = end_pos[0],end_pos[1],end_pos[2]
        vxT, vyT, vzT = end_vel[0],end_vel[1],end_vel[2]

        self._coeff_x = self._compute_3rd_order_polynomial_coeff(r0=x0,
                                                                    rT=xT,
                                                                    rdot0=vx0,
                                                                    rdotT=vxT,
                                                                    T=T)
        
        self._coeff_y = self._compute_3rd_order_polynomial_coeff(r0=y0,
                                                                    rT=yT,
                                                                    rdot0=vy0,
                                                                    rdotT=vyT,
                                                                    T=T)
        
        self._coeff_z = self._compute_3rd_order_polynomial_coeff(r0=z0,
                                                                    rT=zT,
                                                                    rdot0=vz0,
                                                                    rdotT=vzT,
                                                                    T=T)
      
        self.t = 0
        self.T = T
    
    def get_trajectory(self):
        if self.t>=self.T:
            self.t = self.T
        x = self._coeff_x[0] + self._coeff_x[1]*self.t + self._coeff_x[2]*self.t**2 + self._coeff_x[3]*self.t**3
        vx = self._coeff_x[1] + 2*self._coeff_x[2]*self.t + 3*self._coeff_x[3]*self.t**2
        ax = 2*self._coeff_x[2] + 6*self._coeff_x[3]*self.t

        y = self._coeff_y[0] + self._coeff_y[1]*self.t + self._coeff_y[2]*self.t**2 + self._coeff_y[3]*self.t**3
        vy = self._coeff_y[1] + 2*self._coeff_y[2]*self.t + 3*self._coeff_y[3]*self.t**2
        ay = 2*self._coeff_y[2] + 6*self._coeff_y[3]*self.t

        z = self._coeff_z[0] + self._coeff_z[1]*self.t + self._coeff_z[2]*self.t**2 + self._coeff_z[3]*self.t**3
        vz = self._coeff_z[1] + 2*self._coeff_z[2]*self.t + 3*self._coeff_z[3]*self.t**2
        az = 2*self._coeff_z[2] + 6*self._coeff_z[3]*self.t


        pos_t = np.array([x,y,z])
        vel_t = np.array([vx,vy,vz])
        acc_t = np.array([ax,vy,vz])
        
        self.t += self.dt

        return pos_t, vel_t, acc_t

    def _compute_3rd_order_polynomial_coeff(self,r0,rT,rdot0,rdotT,T):
        A = np.array([[1,0,0,0],[1, T, T**2, T**3],[0,1,0,0],[0,1,2*T,3*T**2]])
        b = np.array([r0,rT,rdot0,rdotT])
        invA = np.linalg.inv(A)
        poly_coeff = invA.dot(b)
        return poly_coeff