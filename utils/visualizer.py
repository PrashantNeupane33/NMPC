import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

base = Path(__file__).parent.parent / "data"

states     = np.loadtxt(base / "states.csv",         delimiter=",")
inputs     = np.loadtxt(base / "computedInputs.csv", delimiter=",")
trajectory = np.loadtxt(base / "trajectory.csv",      delimiter=",")

def to_NxN(mat, n_dims):
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)
    if mat.shape[0] != n_dims:
        mat = mat.T
    return mat

states = to_NxN(states, n_dims=3)
inputs = to_NxN(inputs, n_dims=2)

if trajectory.ndim == 1:
    trajectory = trajectory.reshape(-1, 1)
if trajectory.shape[1] != 3:
    trajectory = trajectory.T

N = states.shape[1]
trajectory = trajectory[:N, :]

plt.figure(figsize=(6,6))
plt.plot(trajectory[:,0], trajectory[:,1], 'r--', linewidth=2, label='Reference')
plt.plot(states[0,:],     states[1,:],     'b-',  linewidth=2, label='EKF Estimated')
plt.scatter(states[0,0],  states[1,0], color='green', s=100, zorder=5, label='Start')
plt.xlabel('X [m]')
plt.ylabel('Y [m]')
plt.title('XY Trajectory')
plt.axis('equal')
plt.legend()
plt.grid(True)

plt.figure(figsize=(8,4))
plt.plot(np.degrees(trajectory[:,2]), 'r--', linewidth=2, label='theta ref')
plt.plot(np.degrees(states[2,:]),     'b-',  linewidth=2, label='theta estimated')
plt.xlabel('Timestep')
plt.ylabel('Heading [deg]')
plt.title('Heading')
plt.legend()
plt.grid(True)

plt.figure(figsize=(8,4))
plt.plot(inputs[0,:], linewidth=2, label='v')
plt.plot(inputs[1,:], linewidth=2, label='omega')
plt.xlabel('Timestep')
plt.ylabel('Command')
plt.title('Control Inputs')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
