import numpy as np
import matplotlib.pyplot as plt

# ── Load
states     = np.loadtxt("data/states.csv",         delimiter=",")
inputs     = np.loadtxt("data/computedInputs.csv", delimiter=",")
trajectory = np.loadtxt("data/trajectory.csv",      delimiter=",")

# ── Normalize states and inputs to (3, N)
def to_3xN(mat, n_dims=3):
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)
    if mat.shape[0] != n_dims:
        mat = mat.T
    return mat

states = to_3xN(states, n_dims=3)
inputs = to_3xN(inputs, n_dims=3)

# trajectory is (N, 3) — rows are timesteps, cols are x, y, theta
if trajectory.ndim == 1:
    trajectory = trajectory.reshape(-1, 1)
if trajectory.shape[1] != 3:
    trajectory = trajectory.T

# trim trajectory to match states length
N = states.shape[1]
trajectory = trajectory[:N, :]

# ── Plot 1: XY Trajectory
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

# ── Plot 2: Heading
plt.figure(figsize=(8,4))
plt.plot(np.degrees(trajectory[:,2]), 'r--', linewidth=2, label='theta ref')
plt.plot(np.degrees(states[2,:]),     'b-',  linewidth=2, label='theta estimated')
plt.xlabel('Timestep')
plt.ylabel('Heading [deg]')
plt.title('Heading')
plt.legend()
plt.grid(True)

# ── Plot 3: Control Inputs
plt.figure(figsize=(8,4))
plt.plot(inputs[0,:], linewidth=2, label='vx')
plt.plot(inputs[1,:], linewidth=2, label='vy')
plt.plot(inputs[2,:], linewidth=2, label='omega')
plt.xlabel('Timestep')
plt.ylabel('Command')
plt.title('Control Inputs')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
