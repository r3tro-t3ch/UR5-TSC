from controller.task_space_objective import *
from controller.task_space_controller import TaskSpaceController
from utils.data_logger import Logger
from env.ur5_env import UR5Env
from utils.trajectory_generator import TrajectoryGenerator

class ArmController:

    def __init__(self, env : UR5Env, args) -> None:
        
        self.env = env
        self.time = 0
        self.dt = self.env.model.opt.timestep

        self.des_pos = args['des_pos']
        self.des_ori_q = args['des_ori_q']
        self.des_ori_euler = quat2euler(self.des_ori_q)

        self.pos_task_mode = args['pos_task_mode']
        self.ori_task_mode = args['ori_task_mode']

        self.pos_task = EEPositionTask(
            self.env,
            w=args['pos_task_weight'],
            Kp_track=args['pos_task_kp_track'],
            Kd_track=args['pos_task_kd_track'],
            Kd_damp=args['pos_task_kd_damp']
        )

        self.ori_task = EEOrientationTask(
            self.env,
            w=args['ori_task_weight'],
            Kp_track=args['ori_task_kp_track'],
            Kd_track=args['ori_task_kd_track'],
            Kd_damp=args['ori_task_kd_damp']
        )

        self.tsc = TaskSpaceController(self.env)

        self.traj_handler = TrajectoryGenerator(self.dt)
        self.traj_handler.reset_trajectory(
            self.env.ee_pos,
            self.des_pos,
            np.zeros(3),
            np.zeros(3),
            args['T']
        )

        self.logger = Logger()

    def get_action(self):
        
        self.traj_pos, vel, acc = self.traj_handler.get_trajectory()

        H_pos, g_pos = self.pos_task.get_cost(
            self.traj_pos,
            vel,
            acc,
            self.pos_task_mode
        )

        H_ori, g_ori = self.ori_task.get_cost(
            self.des_ori_q,
            np.zeros(3),
            np.zeros(3),
            self.ori_task_mode
        )

        H = H_pos + H_ori
        g = g_pos + g_ori

        q_ddot, tau = self.tsc.get_action(g_pos, H_pos)

        self.time += self.dt

        return q_ddot, tau
    
    def _log_data(self):

        data = {
            'time' : self.time,

            'ee_pos_x' : self.env.ee_pos[0],
            'ee_pos_y' : self.env.ee_pos[1],
            'ee_pos_z' : self.env.ee_pos[2],

            'ee_pos_x_ref' : self.traj_pos[0],
            'ee_pos_y_ref' : self.traj_pos[1],
            'ee_pos_z_ref' : self.traj_pos[2],

            'ee_ori_x_ref'  : self.des_ori_euler[0],
            'ee_ori_y_ref'  : self.des_ori_euler[1],
            'ee_ori_z_ref'  : self.des_ori_euler[2],

            'ee_ori_x'  : self.env.ee_euler[0],
            'ee_ori_y'  : self.env.ee_euler[1],
            'ee_ori_z'  : self.env.ee_euler[2],

        }

        self.logger.log_data(**data)
    




