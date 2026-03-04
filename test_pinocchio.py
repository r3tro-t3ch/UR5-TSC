import pinocchio
import numpy as np

model = pinocchio.buildModelFromMJCF("./env/ur5e.xml")

print("Model : ", model.nq, model.nv)

data = model.createData()

q   = np.ones((6,)) * 1.57

data.q = q

print("Data : ", data.M)

# model.initViewer(loadModel=True)

J = pinocchio.computeJointJacobian(model, data, q, model.njoints - 1)

print(J)