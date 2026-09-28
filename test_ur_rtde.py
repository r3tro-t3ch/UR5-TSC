# enable ur cap
# https://github.com/UniversalRobots/Universal_Robots_ROS_Driver/blob/master/ur_robot_driver/doc/install_urcap_e_series.md

# basic use
# https://sdurobotics.gitlab.io/ur_rtde/examples/examples.html#basic-use

# import  rtde_receive, rtde_io
# from rtde_control import RTDEControlInterface as RTDEControl

# # rtde_frequency = 500.0
# # rtde_c = RTDEControl("192.168.1.100", rtde_frequency, RTDEControl.FLAG_USE_EXT_UR_CAP)
# # rtde_c.moveL([-0.143, -0.435, 0.20, -0.001, 3.12, 0.04], 0.5, 0.3)

# # rtde_r = rtde_receive.RTDEReceiveInterface("192.168.1.100")
# # actual_q = rtde_r.getActualQ()
# # print("q : ", actual_q)

# from rtde_receive import RTDEReceiveInterface

# rtde_r = RTDEReceiveInterface("192.168.1.100")
# print(rtde_r.getActualTCPPose())

from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
import time

rtde_c = RTDEControlInterface("192.168.1.100")
rtde_r = RTDEReceiveInterface("192.168.1.100")

# Get current pose
current_pose = rtde_r.getActualTCPPose()
print(f"Current pose: {current_pose}")

# Move 5 cm up in Z
target_pose = current_pose.copy()
target_pose[2] += 0.3

# Execute linear motion
rtde_c.moveL(target_pose, speed=0.25, acceleration=0.5, asynchronous=True)

# Wait for movement to complete
time.sleep(0.5)
while abs(rtde_r.getActualTCPSpeed()[2]) > 0.01:
    current_pose = rtde_r.getActualTCPPose()
    print(f"Current pose: {current_pose}")
    time.sleep(0.01)

print("Movement complete")
rtde_c.stopScript()