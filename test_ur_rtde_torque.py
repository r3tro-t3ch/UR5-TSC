import rtde_control
import rtde_receive
import numpy as np
import time

ROBOT_IP = "192.168.1.100"

rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)

dt = 1.0 / 500

print("Starting controlled torque motion...")

try:
    # Break static equilibrium
    # rtde_c.speedJ([0.15, 0, 0, 0, 0, 0], 0.3, 1.0)
    time.sleep(1.0)
    # rtde_c.speedStop()

    MAX_TAU = 25.0

    for i in range(3000):
        start = rtde_c.initPeriod()

        qd = rtde_r.getActualQd()

        # --- CONTROL LAW ---
        # accelerate continuously instead of constant torque
        desired_accel = 0.8  # rad/s^2

        # damping (important for stability)
        tau = MAX_TAU * desired_accel - 3.0 * qd[0]

        # clamp
        tau = np.clip(tau, -MAX_TAU, MAX_TAU)

        torques = [tau, 0, 0, 0, 0, 0]

        rtde_c.directTorque(torques, friction_comp=False)
        rtde_c.waitPeriod(start)

finally:
    print("Stopping...")
    rtde_c.stopScript()
    rtde_c.disconnect()