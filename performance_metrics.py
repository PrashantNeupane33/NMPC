import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────────
# 1. LOAD CSV FILES  (no headers, fixed 0.05s timestep)
# ─────────────────────────────────────────────
states = np.loadtxt("data/states.csv",         delimiter=",")
ref    = np.loadtxt("data/trajectory.csv",     delimiter=",")
inputs = np.loadtxt("data/computedInputs.csv", delimiter=",")

# Trim to same length in case files differ by a row
N  = min(len(states), len(ref), len(inputs))
states = states[:N]
ref    = ref[:N]
inputs = inputs[:N]

dt   = 0.05
time = np.arange(N) * dt   # time axis in seconds

# ─────────────────────────────────────────────
# 2. COMPUTE ERROR SIGNALS (time series)
# ─────────────────────────────────────────────

# --- Position error (Euclidean distance at each step)
dx          = states[:, 0] - ref[:, 0]
dy          = states[:, 1] - ref[:, 1]
pos_error   = np.sqrt(dx**2 + dy**2)

# --- Heading error (wrapped to (-pi, pi])
heading_err = np.arctan2(np.sin(states[:, 2] - ref[:, 2]),
                         np.cos(states[:, 2] - ref[:, 2]))

# --- Cross-track error (signed lateral deviation from path)
cte         = -dx * np.sin(ref[:, 2]) + dy * np.cos(ref[:, 2])

# --- Along-track error (signed longitudinal deviation along path)
along_track =  dx * np.cos(ref[:, 2]) + dy * np.sin(ref[:, 2])

# --- Control effort per step (norm of input vector)
ctrl_effort = np.sqrt(np.sum(inputs**2, axis=1))

# ─────────────────────────────────────────────
# 3. SUMMARY METRICS (single values)
# ─────────────────────────────────────────────
metrics = {
    "Position RMSE (m)"         : np.sqrt(np.mean(pos_error**2)),
    "Position MAE (m)"          : np.mean(pos_error),
    "Max Position Error (m)"    : np.max(pos_error),
    "Heading RMSE (rad)"        : np.sqrt(np.mean(heading_err**2)),
    "Heading MAE (rad)"         : np.mean(np.abs(heading_err)),
    "Max Heading Error (rad)"   : np.max(np.abs(heading_err)),
    "CTE RMSE (m)"              : np.sqrt(np.mean(cte**2)),
    "Max CTE (m)"               : np.max(np.abs(cte)),
    "Along-Track RMSE (m)"      : np.sqrt(np.mean(along_track**2)),
    "Control Effort RMS"        : np.sqrt(np.mean(ctrl_effort**2)),
}

print("\n========= Performance Metrics =========")
for name, val in metrics.items():
    print(f"  {name:<30s}: {val:.4f}")
print("=======================================\n")

# ─────────────────────────────────────────────
# 4. PLOTS
# ─────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor" : "white",
    "axes.facecolor"   : "white",
    "axes.grid"        : True,
    "grid.alpha"       : 0.4,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "font.size"        : 11,
})

# ── Figure 1: XY Trajectory comparison ──────
fig1, ax = plt.subplots(figsize=(7, 6))
ax.plot(ref[:, 0],    ref[:, 1],    "k--",  lw=1.5, label="Reference")
ax.plot(states[:, 0], states[:, 1], "b-",   lw=1.5, label="Actual")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title("XY Trajectory: Reference vs Actual")
ax.legend()
ax.set_aspect("equal")
plt.tight_layout()

# ── Figure 2: Error time series (2×2 grid) ──
fig2, axes = plt.subplots(2, 2, figsize=(12, 7))
fig2.suptitle("Tracking Error Over Time", fontsize=13, fontweight="bold")

# Position error
axes[0, 0].plot(time, pos_error, color="steelblue", lw=1.2)
axes[0, 0].axhline(metrics["Position RMSE (m)"], color="red",
                   ls="--", lw=1, label=f"RMSE = {metrics['Position RMSE (m)']:.4f} m")
axes[0, 0].set_title("Position Error")
axes[0, 0].set_ylabel("Error (m)")
axes[0, 0].set_xlabel("Time (s)")
axes[0, 0].legend(fontsize=9)

# Heading error
axes[0, 1].plot(time, np.degrees(heading_err), color="darkorange", lw=1.2)
axes[0, 1].axhline(np.degrees(metrics["Heading RMSE (rad)"]), color="red",
                   ls="--", lw=1,
                   label=f"RMSE = {np.degrees(metrics['Heading RMSE (rad)']):.3f}°")
axes[0, 1].set_title("Heading Error")
axes[0, 1].set_ylabel("Error (deg)")
axes[0, 1].set_xlabel("Time (s)")
axes[0, 1].legend(fontsize=9)

# Cross-track error
axes[1, 0].plot(time, cte, color="seagreen", lw=1.2)
axes[1, 0].axhline(0, color="black", lw=0.8, ls="-")
axes[1, 0].axhline( metrics["CTE RMSE (m)"], color="red",
                    ls="--", lw=1, label=f"RMSE = {metrics['CTE RMSE (m)']:.4f} m")
axes[1, 0].axhline(-metrics["CTE RMSE (m)"], color="red", ls="--", lw=1)
axes[1, 0].set_title("Cross-Track Error (CTE)")
axes[1, 0].set_ylabel("CTE (m)")
axes[1, 0].set_xlabel("Time (s)")
axes[1, 0].legend(fontsize=9)

# Along-track error
axes[1, 1].plot(time, along_track, color="mediumpurple", lw=1.2)
axes[1, 1].axhline(0, color="black", lw=0.8, ls="-")
axes[1, 1].axhline( metrics["Along-Track RMSE (m)"], color="red",
                    ls="--", lw=1,
                    label=f"RMSE = {metrics['Along-Track RMSE (m)']:.4f} m")
axes[1, 1].axhline(-metrics["Along-Track RMSE (m)"], color="red", ls="--", lw=1)
axes[1, 1].set_title("Along-Track Error")
axes[1, 1].set_ylabel("Error (m)")
axes[1, 1].set_xlabel("Time (s)")
axes[1, 1].legend(fontsize=9)

plt.tight_layout()

# ── Figure 3: Control inputs over time ──────
fig3, axes3 = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
fig3.suptitle("Control Inputs Over Time", fontsize=13, fontweight="bold")

labels = ["$v_x$ (m/s)", "$v_y$ (m/s)", "$\\omega$ (rad/s)"]
colors = ["steelblue", "darkorange", "seagreen"]
for i in range(3):
    axes3[i].plot(time, inputs[:, i], color=colors[i], lw=1.2)
    axes3[i].set_ylabel(labels[i])
    axes3[i].axhline(0, color="black", lw=0.6, ls="--")
axes3[2].set_xlabel("Time (s)")
plt.tight_layout()

# ── Figure 4: Summary metrics bar chart ─────
fig4, ax4 = plt.subplots(figsize=(9, 5))
bar_names = [
    "Pos RMSE\n(m)",
    "Pos MAE\n(m)",
    "Max Pos\nError (m)",
    "Heading\nRMSE (rad)",
    "Heading\nMAE (rad)",
    "Max Heading\nError (rad)",
    "CTE\nRMSE (m)",
    "Max CTE\n(m)",
    "Along-Track\nRMSE (m)",
    "Ctrl Effort\nRMS",
]
bar_vals = list(metrics.values())
bar_colors = [
    "steelblue", "cornflowerblue", "lightsteelblue",
    "darkorange", "sandybrown", "peachpuff",
    "seagreen", "mediumseagreen",
    "mediumpurple",
    "tomato",
]
bars = ax4.bar(bar_names, bar_vals, color=bar_colors, edgecolor="white", width=0.6)
for bar, val in zip(bars, bar_vals):
    ax4.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + max(bar_vals) * 0.01,
             f"{val:.4f}", ha="center", va="bottom", fontsize=8)
ax4.set_title("Summary Performance Metrics")
ax4.set_ylabel("Value")
plt.tight_layout()

plt.show()
