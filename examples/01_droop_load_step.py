"""
Example 01 – REGFM_A1: Droop response to a load step
=====================================================

Scenario
--------
The inverter starts at P_ref = 0.3 pu.  At t = 1 s the load steps to
0.7 pu.  We observe:
  - instantaneous frequency jump (droop has no inertia)
  - fast power ramp to new setpoint
  - new steady-state frequency according to the droop characteristic:
      Δf = −kp · ΔP · f₀ = −0.05 × 0.4 × 60 = −1.2 Hz
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt

from gfm.models.regfm_a1 import REGFM_A1, DroopParams
from gfm.models.base import GridParams, OMEGA0
from simulations.runner import simulate


# ------------------------------------------------------------------
# Model and grid setup
# ------------------------------------------------------------------
grid = GridParams(Xg=0.3, Vg=1.0)
params = DroopParams(kp=0.05, kq=0.05, tau_p=0.05, tau_q=0.05)
model = REGFM_A1(params=params, grid=grid)

# Time-varying load: step at t=1 s
P_BEFORE, P_AFTER = 0.3, 0.7
T_STEP = 1.0

def P_ref(t):
    return P_AFTER if t >= T_STEP else P_BEFORE

# ------------------------------------------------------------------
# Simulate
# ------------------------------------------------------------------
result = simulate(
    model,
    t_span=(0.0, 4.0),
    P_ref=P_ref,
    Q_ref=0.0,
    omega_grid=OMEGA0,
    x0=model.x0(P_BEFORE, 0.0),
    n_eval=4000,
)

# Theoretical steady-state frequency deviation
df_ss = -params.kp * (P_AFTER - P_BEFORE) * 60.0
print(f"Expected Δf (droop) = {df_ss:.3f} Hz")
print(f"Simulated final f   = {result.get('freq_hz')[-1]:.3f} Hz")
print(f"Power ripple        = {result.P_ripple_pu():.4f} pu")

# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------
t = result.t
P = result.get("P_pu")
f = result.get("freq_hz")
E = result.get("E_pu")
d = result.get("delta_deg")

fig, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
fig.suptitle("REGFM_A1 (Droop) – Load Step Response", fontsize=14, fontweight="bold")

axes[0].plot(t, result.P_ref_vec, "k--", lw=1.2, label="P_ref")
axes[0].plot(t, P, lw=2, label="P_out")
axes[0].set_ylabel("Active power [pu]")
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim([0.0, 1.0])

axes[1].plot(t, f, lw=2, color="C1")
axes[1].axhline(60.0 + df_ss, ls="--", color="gray", lw=1, label=f"SS target {60+df_ss:.2f} Hz")
axes[1].set_ylabel("Frequency [Hz]")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].plot(t, E, lw=2, color="C2")
axes[2].set_ylabel("Voltage |E| [pu]")
axes[2].grid(True, alpha=0.3)

axes[3].plot(t, d, lw=2, color="C3")
axes[3].set_ylabel("Power angle δ [°]")
axes[3].set_xlabel("Time [s]")
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("01_droop_load_step.png", dpi=150)
print("Saved 01_droop_load_step.png")
plt.show()
