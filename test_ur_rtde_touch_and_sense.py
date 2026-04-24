# test from https://sdurobotics.gitlab.io/ur_rtde/api/api.html#_CPPv4N7ur_rtde20RTDEReceiveInterface24getActualCurrentAsTorqueEv


# Recieve interface
# getActualCurrentAsTorque()
# getFtRawWrench()
# getActualTCPForce()

# control interface
# getJointTorques()
# startContactDetection()
# readContactDetection()
# stopContactDetection()
# getJacobain()
# getJacobianTimeDerivative
# bool moveUntilContact(const std::vector<double> &xd, const std::vector<double> &direction = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, double acceleration = 0.5)


from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
import time

rtde_c = RTDEControlInterface("192.168.1.100")
rtde_r = RTDEReceiveInterface("192.168.1.100")

# get current ee pose
pose = rtde_r.getActualTCPPose()

# pose right end - Pose :  [0.28740747729998417, 0.8019096329869324, 0.3230281903918789, 0.0451976397077111, -2.1533954936290836, -2.1784652968742417]
# pose left end -  Pose :  [-0.22680639826723198, 0.7845722206754894, 0.32134735418864513, 0.00010496402434214192, -2.1825599995570815, -2.149597862044473]

left_end_pose = [-0.22680639826723198, 0.7845722206754894, 0.32134735418864513, 0.00010496402434214192, -2.1825599995570815, -2.149597862044473]
right_end_pose = [0.28740747729998417, 0.8019096329869324, 0.3230281903918789, 0.0451976397077111, -2.1533954936290836, -2.1784652968742417]

start_pose = [-0.0052165730680607847, 0.344666335312934, 0.38547124717109255, 0.024220490453556575, -2.148725145260427, -2.088536676723097]

# right_end_back_pose = []



# print("Pose : ", pose)


touch_poses = []

# get ee wrench
# ee_wrench = rtde_r.getActualTCPForce()
# ()

# print("Wrench : ", ee_wrench)

time.sleep(5)

# move until touch
rtde_c.moveL(start_pose)

rtde_c.moveL(left_end_pose)

contact = rtde_c.moveUntilContact(
    [0, 0, -0.005, 0, 0, 0]
)

print("Wrench on contact : ", rtde_r.getActualTCPForce())
time.sleep(1)

touch_poses.append(rtde_r.getActualTCPPose())

rtde_c.moveL(left_end_pose)

contact = rtde_c.moveUntilContact(
    [0, 0.005, 0, 0, 0, 0]
)
print("Wrench on contact : ", rtde_r.getActualTCPForce())
time.sleep(1)

touch_poses.append(rtde_r.getActualTCPPose())

rtde_c.moveL(left_end_pose)

rtde_c.moveL(right_end_pose)

contact = rtde_c.moveUntilContact(
    [0, 0, -0.005, 0, 0, 0]
)

print("Wrench on contact : ", rtde_r.getActualTCPForce())
time.sleep(1)


touch_poses.append(rtde_r.getActualTCPPose())

rtde_c.moveL(right_end_pose)

contact = rtde_c.moveUntilContact(
    [0, 0.005, 0, 0, 0, 0]
)
print("Wrench on contact : ", rtde_r.getActualTCPForce())
time.sleep(1)

touch_poses.append(rtde_r.getActualTCPPose())

rtde_c.moveL(right_end_pose)

rtde_c.moveL(start_pose)


print("Pose on contact : ", touch_poses)

# output
# Pose on contact :  
# [
# [-0.2268035981528113, 0.7845826671967524, 0.3156868889131381, 9.875287990248081e-05, -2.1824582587356707, -2.149695730556573],
# [-0.2268414648170323, 0.802205506308417, 0.3213441044930284, 0.00012926451952690948, -2.1825334563453027, -2.1495730464941913],
# [0.2874131929151962, 0.8019049072526326, 0.3122546543631593, 0.04512811605328812, -2.153392207831968, -2.178473505790722], 
# [0.2874211680896507, 0.8082524008519458, 0.3230183318756137, 0.045167528664406424, -2.153386435540462, -2.178462763669421]
# ]