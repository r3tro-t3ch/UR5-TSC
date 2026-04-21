# enable ur cap
# https://github.com/UniversalRobots/Universal_Robots_ROS_Driver/blob/master/ur_robot_driver/doc/install_urcap_e_series.md

# basic use
# https://sdurobotics.gitlab.io/ur_rtde/examples/examples.html#basic-use

import  rtde_receive, rtde_io
from rtde_control import RTDEControlInterface as RTDEControl

rtde_frequency = 500.0
rtde_c = RTDEControl("127.0.0.1", rtde_frequency, RTDEControl.FLAG_USE_EXT_UR_CAP)
rtde_c.moveL([-0.143, -0.435, 0.20, -0.001, 3.12, 0.04], 0.5, 0.3)

rtde_r = rtde_receive.RTDEReceiveInterface("127.0.0.1")
actual_q = rtde_r.getActualQ()
print("q : ", actual_q)

