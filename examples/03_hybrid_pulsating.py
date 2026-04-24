"""
Example 03 – REGFM_C1: Hybrid GFM vs pulsating AI / data-centre loads
======================================================================

Scenario
--------
A multi-component pulsating load representative of AI/data-centre workloads
is applied to all three GFM topologies:

    P_ref(t) = 0.6 + 0.10·sin(2π·2t) + 0.07·sin(2π·8t) + 0.04·sin(2π·22t)

Frequency range: 2–22 Hz — squarely in the REGFM_C1 design window.

Expected results
----------------
* A1 (Droop)  : large power ripple, significant frequency oscillation
* B1 (VSM)    : VSM inertia partially attenuates low-end, amplifies mid-range
* C1 (Hybrid) : feedforward path suppresses 2–22 Hz; much smaller Δf and ΔP
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt

from gfm.models.regfm_a1 import REGFM_A1, DroopParams
from gfm.models.regfm_b1 import REGFM_B1, VSMParams
from gfm.models.regfm_c1 import REGFM_C1, HybridParams
from gfm.models.base import GridParams, OMEGA0
from gfm.loads.pulsating import PulsatingLoad
from simulations.runner import simulate


# ------------------------------------------------------------------
# Load profile
# ------------------------------------------------------------------
load = PulsatingLoad.ai_datacenter(P_base=0.6, t_start=0.3)

# ------------------------------------------------------------------
# Models
# ------------------------------------------------------------------
grid = GridParams(Xg=0.3, Vg=1.0)

a1 = REGFM_A1(DroopParams(kp=0.05, kq=0.05, tau_p=0.05), grid)
b1 = REGFM_B1(VSMParams(H=5.0, Dp=10.0, kq=0.05, tau_p=0.05), grid)
c1 = REGFM_C1(HybridParams(H=5.0, Dp=10.0, kq=0.05, kff=0.10,
                             tau_fast=0.005, tau_slow=0.10), grid)

T_END = 5.0
P0 = 0.6

# ------------------------------------------------------------------
# Simulate
# ------------------------------------------------------------------
res_a1 = simulate(a1, (0.0, T_END), P_ref=load.P_ref, x0=a1.x0(P0))
res_b1 = simulate(b1, (0.0, T_END), P_ref=load.P_ref, x0=b1.x0(P0))
res_c1 = simulate(c1, (0.0, T_END), P_ref=load.P_ref, x0=c1.x0(P0))

# ------------------------------------------------------------------
# Print metrics
# ------------------------------------------------------------------
print(f"\n{'Metric':<30} {'A1 Droop':>12} {'B1 VSM':>12} {'C1 Hybrid':>12}")
print("─" * 66)
metrics = [
    ("P ripple [pu]",   lambda r: f"{r.P_ripple_pu():.4f}"),
    ("f nadir [Hz]",    lambda r: f"{r.freq_nadir_hz():.4f}"),
    ("f peak [Hz]",     lambda r: f"{r.freq_peak_hz():.4f}"),
    ("RoCoF max [Hz/s]",lambda r: f"{r.RoCoF_max():.3f}"),
]
for name, fn in metrics:
    print(f"{name:<30} {fn(res_a1):>12} {fn(res_b1):>12} {fn(res_c1):>12}")

improvement_p = (1 - res_c1.P_ripple_pu() / res_a1.P_ripple_pu()) * 100
print(f"\nC1 reduces P-ripple by {improvement_p:.1f}% vs A1")

# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
fig.suptitle(
    "REGFM_C1 Hybrid vs A1 Droop & B1 VSM – Pulsating AI Load (0.3–22 Hz)",
    fontsize=12, fontweight="bold"
)

labels = ["A1 Droop", "B1 VSM", "C1 Hybrid"]
results = [res_a1, res_b1, res_c1]
colors = ["#E63946", "#457B9D", "#2A9D8F"]
lws = [1.5, 1.5, 2.5]

# --- P_ref overlay ---
axes[0].plot(res_a1.t, res_a1.P_ref_vec, "k--", lw=1, alpha=0.6, label="P_ref")
for res, lbl, col, lw in zip(results, labels, colors, lws):
    axes[0].plot(res.t, res.get("P_pu"), lw=lw, color=col, label=lbl)
axes[0].set_ylabel("Active power [pu]")
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

for res, lbl, col, lw in zip(results, labels, colors, lws):
    axes[1].plot(res.t, res.get("freq_dev_hz"), lw=lw, color=col, label=lbl)
axes[1].axhline(0, ls="--", color="gray", lw=1)
axes[1].set_ylabel("Frequency deviation [Hz]")
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

for res, lbl, col, lw in zip(results, labels, colors, lws):
    axes[2].plot(res.t, res.get("E_pu"), lw=lw, color=col, label=lbl)
axes[2].set_ylabel("Voltage |E| [pu]")
axes[2].set_xlabel("Time [s]")
axes[2].legend(fontsize=9)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("03_hybrid_pulsating.png", dpi=150)
print("\nSaved 03_hybrid_pulsating.png")
plt.show()
