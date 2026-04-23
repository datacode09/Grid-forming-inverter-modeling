"""
Example 02 – REGFM_B1: VSM synthetic inertia comparison
=========================================================

Scenario
--------
A large load step (0.3 → 0.7 pu at t = 1 s) is applied to both the
droop-only model (A1) and the VSM model (B1).

Key observations
----------------
* B1 has a lower frequency nadir (slower initial drop) due to virtual inertia
  smoothing the RoCoF.
* B1 takes longer to reach steady state but protects against fast disturbances.
* Both converge to the same droop-determined steady-state frequency because B1
  also has an equivalent P-ω droop embedded in its damping term.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt

from gfm.models.regfm_a1 import REGFM_A1, DroopParams
from gfm.models.regfm_b1 import REGFM_B1, VSMParams
from gfm.models.base import GridParams, OMEGA0
from simulations.runner import simulate


# ------------------------------------------------------------------
# Shared scenario
# ------------------------------------------------------------------
grid = GridParams(Xg=0.3, Vg=1.0)
P_BEFORE, P_AFTER = 0.3, 0.7
T_STEP = 1.0
T_END = 8.0

def P_ref(t):
    return P_AFTER if t >= T_STEP else P_BEFORE

# ------------------------------------------------------------------
# Models
# ------------------------------------------------------------------
droop = REGFM_A1(params=DroopParams(kp=0.05, kq=0.05, tau_p=0.05), grid=grid)

vsm_configs = {
    "B1 H=2 s": REGFM_B1(VSMParams(H=2.0, Dp=10.0, kq=0.05), grid),
    "B1 H=5 s": REGFM_B1(VSMParams(H=5.0, Dp=10.0, kq=0.05), grid),
    "B1 H=10 s": REGFM_B1(VSMParams(H=10.0, Dp=10.0, kq=0.05), grid),
}

# ------------------------------------------------------------------
# Simulate
# ------------------------------------------------------------------
res_droop = simulate(droop, (0.0, T_END), P_ref=P_ref, x0=droop.x0(P_BEFORE))

vsm_results = {}
for label, mdl in vsm_configs.items():
    vsm_results[label] = simulate(mdl, (0.0, T_END), P_ref=P_ref, x0=mdl.x0(P_BEFORE))

# ------------------------------------------------------------------
# Print summary
# ------------------------------------------------------------------
print(f"{'Model':<20} {'f_nadir [Hz]':>14} {'RoCoF_max [Hz/s]':>18}")
print("-" * 54)
print(f"{'A1 Droop':<20} {res_droop.freq_nadir_hz():>14.3f} {res_droop.RoCoF_max():>18.3f}")
for label, res in vsm_results.items():
    print(f"{label:<20} {res.freq_nadir_hz():>14.3f} {res.RoCoF_max():>18.3f}")

# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
fig.suptitle("REGFM_A1 vs REGFM_B1 – Synthetic Inertia Comparison", fontsize=13, fontweight="bold")

colors = ["C0", "C1", "C2", "C3"]

for ax in axes:
    ax.axvline(T_STEP, ls=":", color="gray", lw=1, alpha=0.7)

# --- Power ---
axes[0].plot(res_droop.t, res_droop.P_ref_vec, "k--", lw=1.2, label="P_ref")
axes[0].plot(res_droop.t, res_droop.get("P_pu"), lw=2, color=colors[0], label="A1 Droop")
for i, (label, res) in enumerate(vsm_results.items(), start=1):
    axes[0].plot(res.t, res.get("P_pu"), lw=2, color=colors[i], label=label)
axes[0].set_ylabel("Active power [pu]")
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

# --- Frequency ---
axes[1].plot(res_droop.t, res_droop.get("freq_hz"), lw=2, color=colors[0], label="A1 Droop")
for i, (label, res) in enumerate(vsm_results.items(), start=1):
    axes[1].plot(res.t, res.get("freq_hz"), lw=2, color=colors[i], label=label)
axes[1].axhline(60.0, ls="--", color="gray", lw=1, alpha=0.5)
axes[1].set_ylabel("Frequency [Hz]")
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

# --- Voltage ---
axes[2].plot(res_droop.t, res_droop.get("E_pu"), lw=2, color=colors[0], label="A1 Droop")
for i, (label, res) in enumerate(vsm_results.items(), start=1):
    axes[2].plot(res.t, res.get("E_pu"), lw=2, color=colors[i], label=label)
axes[2].set_ylabel("Voltage |E| [pu]")
axes[2].set_xlabel("Time [s]")
axes[2].legend(fontsize=8)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("02_vsm_inertia.png", dpi=150)
print("Saved 02_vsm_inertia.png")
plt.show()
