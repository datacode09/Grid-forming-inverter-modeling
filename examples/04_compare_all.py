"""
Example 04 – Full comparison: REGFM_A1 / B1 / C1 across three scenarios
=========================================================================

Scenarios
---------
1. Load step (0.3 → 0.7 pu at t = 1 s)
2. Single-frequency pulsation at 5 Hz
3. Single-frequency pulsation at 20 Hz

For each scenario the active power, frequency deviation, and voltage are
compared across all three GFM control architectures.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from gfm.models.regfm_a1 import REGFM_A1, DroopParams
from gfm.models.regfm_b1 import REGFM_B1, VSMParams
from gfm.models.regfm_c1 import REGFM_C1, HybridParams
from gfm.models.base import GridParams, OMEGA0
from simulations.runner import simulate


# ------------------------------------------------------------------
# Build models (shared grid)
# ------------------------------------------------------------------
grid = GridParams(Xg=0.3, Vg=1.0)

models = {
    "A1 Droop": REGFM_A1(DroopParams(kp=0.05, kq=0.05, tau_p=0.05), grid),
    "B1 VSM":   REGFM_B1(VSMParams(H=5.0, Dp=10.0, kq=0.05, tau_p=0.05), grid),
    "C1 Hybrid": REGFM_C1(HybridParams(H=5.0, Dp=10.0, kq=0.05,
                                        kff=0.10, tau_fast=0.005,
                                        tau_slow=0.10), grid),
}
COLORS = {"A1 Droop": "#E63946", "B1 VSM": "#457B9D", "C1 Hybrid": "#2A9D8F"}
LW = {"A1 Droop": 1.5, "B1 VSM": 1.5, "C1 Hybrid": 2.5}

# ------------------------------------------------------------------
# Define scenarios
# ------------------------------------------------------------------
def make_step(P0=0.3, P1=0.7, t_step=1.0):
    def _P(t):
        return P1 if t >= t_step else P0
    return _P, P0, (0.0, 6.0)

def make_pulse(P_base=0.5, amp=0.15, freq_hz=5.0):
    def _P(t):
        return P_base + amp * np.sin(2.0 * np.pi * freq_hz * t)
    return _P, P_base, (0.0, 3.0)

scenarios = {
    "Load Step (0.3→0.7 pu)": make_step(),
    "Pulsation @ 5 Hz":        make_pulse(amp=0.15, freq_hz=5.0),
    "Pulsation @ 20 Hz":       make_pulse(amp=0.10, freq_hz=20.0),
}

# ------------------------------------------------------------------
# Run all simulations
# ------------------------------------------------------------------
all_results = {}   # {scenario: {model_name: SimResult}}

for sc_name, (P_fn, P0, t_span) in scenarios.items():
    all_results[sc_name] = {}
    for m_name, mdl in models.items():
        res = simulate(
            mdl,
            t_span=t_span,
            P_ref=P_fn,
            x0=mdl.x0(P0),
            n_eval=3000,
        )
        all_results[sc_name][m_name] = res

# ------------------------------------------------------------------
# Summary table
# ------------------------------------------------------------------
print(f"\n{'Scenario':<30} {'Model':<14} {'ΔP_pp [pu]':>12} {'f_dev_max [Hz]':>16} {'RoCoF [Hz/s]':>14}")
print("─" * 88)
for sc_name, model_results in all_results.items():
    for m_name, res in model_results.items():
        f_dev = res.get("freq_dev_hz")
        f_max = float(np.max(np.abs(f_dev)))
        print(
            f"{sc_name:<30} {m_name:<14} "
            f"{res.P_ripple_pu():>12.4f} {f_max:>16.4f} {res.RoCoF_max():>14.3f}"
        )
    print()

# ------------------------------------------------------------------
# Plot: 3 scenarios × 2 metrics (P, freq_dev) = 3×2 grid
# ------------------------------------------------------------------
n_sc = len(scenarios)
fig = plt.figure(figsize=(15, 11))
gs = gridspec.GridSpec(2, n_sc, hspace=0.45, wspace=0.35)
fig.suptitle(
    "GFM Control Comparison: REGFM A1 / B1 / C1",
    fontsize=14, fontweight="bold", y=0.98
)

for col, (sc_name, model_results) in enumerate(all_results.items()):
    # ---- Active power ----
    ax_p = fig.add_subplot(gs[0, col])
    first_res = next(iter(model_results.values()))
    ax_p.plot(first_res.t, first_res.P_ref_vec, "k--", lw=1.0, alpha=0.5, label="P_ref")
    for m_name, res in model_results.items():
        ax_p.plot(
            res.t, res.get("P_pu"),
            lw=LW[m_name], color=COLORS[m_name], label=m_name
        )
    ax_p.set_title(sc_name, fontsize=9, pad=4)
    ax_p.set_ylabel("P [pu]" if col == 0 else "")
    ax_p.legend(fontsize=7, loc="upper right")
    ax_p.grid(True, alpha=0.25)
    ax_p.set_ylim([0.0, 1.0])

    # ---- Frequency deviation ----
    ax_f = fig.add_subplot(gs[1, col])
    for m_name, res in model_results.items():
        ax_f.plot(
            res.t, res.get("freq_dev_hz"),
            lw=LW[m_name], color=COLORS[m_name], label=m_name
        )
    ax_f.axhline(0, ls="--", color="gray", lw=0.8)
    ax_f.set_ylabel("Δf [Hz]" if col == 0 else "")
    ax_f.set_xlabel("Time [s]")
    ax_f.legend(fontsize=7, loc="upper right")
    ax_f.grid(True, alpha=0.25)

plt.savefig("04_compare_all.png", dpi=150, bbox_inches="tight")
print("Saved 04_compare_all.png")
plt.show()
