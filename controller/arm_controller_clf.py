from controller.task_space_objective import *
from controller.tsc_clf import CLFTaskSpaceController
from utils.data_logger import Logger
from env.ur_env import UR10eEnv
from utils.trajectory_generator import TrajectoryGenerator, TrajectoryGenerator3rdOrderMidPoint

class CLFArmController:

    def __init__(self, env : UR10eEnv, args) -> None:
        
        self.env = env
        self.time = 0
        self.dt = self.env.model.opt.timestep

        self.des_pos        = args['des_pos']
        self.des_ori_q      = args['des_ori_q']
        self.des_ori_euler  = quat2euler(self.des_ori_q)

        self.T              = args['T']

        self.P              = args['P']
        self.D              = args['D']

        self.init           = True

        self.alpha      = args['alpha']
        self.tau_max    = args['tau_max']

        self.tau        = np.zeros((self.env.model.nu,))

        self.tsc = CLFTaskSpaceController(
            env,
            self.alpha,
            self.P,
            self.D
        )

        self.task = TaskConsistantEETask(
            self.env,
            Kp_track_pos=args['position_task_kp_track'],
            Kd_track_pos=args['position_task_kd_track'],
            Kd_damp_pos=args['position_task_kd_damp'],
            Kp_track_ori=args['orientation_task_kp_track'],
            Kd_track_ori=args['orientation_task_kd_track'],
            Kd_damp_ori=args['orientation_task_kd_damp']
        )

        self.traj_handler = TrajectoryGenerator(self.dt)
        
        self.logger = Logger()

    def get_action(self):

        if self.init:
            self.traj_handler.reset_trajectory(
                self.env.ee_pos,
                self.des_pos,
                np.zeros(3),
                np.zeros(3),
                self.T
            )
            self.init = False
        
        self.traj_pos, vel, acc = self.traj_handler.get_trajectory()

        self.traj_vel = vel

        f_d = self.task.get_cost(
            self.traj_pos,
            vel,
            acc,
            self.des_ori_q,
            np.zeros(3),
            np.zeros(3),
            "track"
        )

        tau = self.tsc.get_action(
            tau_max=self.tau_max,
            x_d=self.traj_pos,
            xdot_d=vel,
            delta_q=self.get_quat_error(self.env.ee_q, self.des_ori_q),
            w_d=np.zeros((3,)),
            x_ddot_d=np.concatenate([acc, np.zeros((3,))]),
            W=np.identity(6),
            f_d=f_d
        )

        self.tau = tau

        self.time += self.dt

        return tau
    
    def get_quat_error(self, q, q_d):
        a = np.array(q_d[1:4])
        b = np.array(q[1:4])
        q_d_x = skew_symmetric(a)
        e = q[0]*a - q_d[0]*b - q_d_x @ b
        return e
    
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

            'tau'       : self.tau,

            'ee_vel_x' : self.env.ee_vel[0],
            'ee_vel_y' : self.env.ee_vel[1],
            'ee_vel_z' : self.env.ee_vel[2],

            'ee_vel_x_ref'  : self.traj_vel[0],
            'ee_vel_y_ref'  : self.traj_vel[1],
            'ee_vel_z_ref'  : self.traj_vel[2],

        }

        self.logger.log_data(**data)
    




