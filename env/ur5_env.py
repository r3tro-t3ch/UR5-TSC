import mujoco as mj
import mujoco_viewer
import numpy as np
import os
from utils.utils import quat2euler

class UR5Env:

    class BodyIndex():

        BASE            = 1
        SHOULDER_LINK   = 2
        UPPER_ARM_LINK  = 3
        FOREARM_LINK    = 4
        WRIST_1_LINK    = 5
        WRIST_2_LINK    = 6
        WRIST_3_LINK    = 7

        EE_SITE         = 0

    def __init__(self, args):
        
        self.is_render      = args['is_render']
        self.xml_file       = args['xml_file']
        xml_directory       = os.getcwd() + "/env"
        self.xml_path       = os.path.join(xml_directory,self.xml_file)
        self.cam_azi        = args['cam_azi']
        self.cam_ele        = args['cam_ele']
        self.cam_dist       = args['cam_dist']
        self._mj_init()             # initialize mujoco data structures
        self.is_alive       = True


        # robot dynamics

        # position and velocity
        self.ee_pos = None
        self.ee_vel = None

        # orientation and angular velocity
        self.ee_q = None
        self.ee_euler = None
        self.ee_w = None

        # position and orientation jacobian
        self.jacp = np.zeros((3, self.model.nv))
        self.jacr = np.zeros((3, self.model.nv))

        # mass matrix, coriolis and gravity vector
        self.M    = np.zeros((self.model.nv,self.model.nv))
        self.C    = np.zeros((self.model.nv,))

        # internal variables
        mj.mj_forward(self.model,self.data)
        
        # update robot states at initialization
        self.update_robot_states()
        
        # robot physical parameters relevant for control
        self.total_mass  = np.sum(self.model.body_mass[1:])

    def update_cntrl(self,torq):
        # torques
        self.data.ctrl = torq

    def step(self,torq=np.zeros((6,))):
        self.update_cntrl(torq)
        mj.mj_step(self.model,self.data)
        if self.is_render:
            self.render()
        self.update_robot_states()
        self.check_if_alive()

    def update_robot_states(self):
        
        # update jacobians
        mj.mj_jacSite(self.model, self.data, self.jacp, self.jacr, self.BodyIndex.EE_SITE)

        # update EOM
        mj.mj_fullM(self.model, self.M, self.data.qM)
        self.C = self.data.qfrc_bias

        # update pos and vel
        self.ee_pos     = self.data.site_xpos[self.BodyIndex.EE_SITE]
        self.ee_vel     = self.jacp @ self.data.qvel

        self.ee_q       = self.data.xquat[self.BodyIndex.WRIST_3_LINK]
        self.ee_euler   = quat2euler(self.ee_q)
        self.ee_w       = self.jacr @ self.data.qvel


    def apply_external_force(self,body,force):
        # apply externam force on any given body
        self.data.xfrc_applied[body,:3] = force

    def check_if_alive(self):
        # self.is_alive = True if (self.torso_zpos > 0.3 and self.torso_zpos < 1.25) else False
        self.is_alive = self.is_alive and self.viewer.is_alive

    def _mj_init(self):
        self.model = mj.MjModel.from_xml_path(self.xml_path)    
        self.data = mj.MjData(self.model) 
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data,hide_menus=False)
        self.viewer.cam.azimuth = self.cam_azi
        self.viewer.cam.elevation = self.cam_ele
        self.viewer.cam.distance =  self.cam_dist


    def render(self):
        self.viewer.render()
            
    def stop(self):
        if self.viewer.is_alive:
            self.viewer.close()

