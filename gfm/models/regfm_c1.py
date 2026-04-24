"""
REGFM_C1 – Hybrid Grid-Forming Control
=======================================

Architecture
------------
REGFM_C1 combines VSM inertia emulation with a high-bandwidth
feedforward (FF) path on the power/current control loop.

    ω_vsm  : slow VSM frequency  (low-pass view, τ_slow ≈ 0.1 s)
    ω_ff   : fast feedforward correction  (τ_fast ≈ 0.005 s)
    ω_total = ω_vsm + ω₀ · kff · (P_ref − Pm_fast)

The FF term acts as a proportional high-bandwidth droop that directly
compensates pulsating power without waiting for the slow VSM rotor
dynamics.  For load oscillations in 0.1–30 Hz, ω_ff suppresses the
power ripple, dramatically reducing frequency and voltage excursions.

State vector  x = [δ, ω, E, Pm_slow, Pm_fast, Qm]
    δ       : power angle  [rad]
    ω       : VSM virtual-rotor angular frequency  [rad/s]
    E       : internal voltage magnitude  [pu]
    Pm_slow : slowly-filtered active power (feeds VSM inertia)  [pu]
    Pm_fast : quickly-filtered active power (feeds FF path)  [pu]
    Qm      : filtered reactive power  [pu]

Design guidelines
-----------------
* kff ≈ 0.05–0.15 ;  larger values improve fast disturbance rejection
  but reduce phase margin — tune with root-locus or Bode analysis.
* τ_fast = 1/(2π·f_BW) ; for 30 Hz bandwidth ≈ 0.005 s.
* τ_slow ≥ 0.1 s keeps VSM acting only on low-frequency dynamics.
"""

from dataclasses import dataclass
import numpy as np

from .base import BaseGFM, GridParams, OMEGA0


@dataclass
class HybridParams:
    # VSM base
    H: float = 5.0          # virtual inertia constant  [s]
    Dp: float = 10.0        # per-unit damping  [pu/pu]
    kq: float = 0.05        # Q–V droop gain  [pu/pu]
    tau_v: float = 0.05     # voltage control time constant  [s]
    tau_q: float = 0.05     # reactive-power filter  [s]
    # Feedforward path
    kff: float = 0.10       # feedforward gain  [pu/pu]
    tau_fast: float = 0.005  # fast power-filter time constant  [s]  (~32 Hz BW)
    tau_slow: float = 0.10   # slow power-filter time constant  [s]  (~1.6 Hz BW)
    E0: float = 1.0          # voltage setpoint  [pu]


class REGFM_C1(BaseGFM):
    """Hybrid GFM inverter with high-bandwidth feedforward (REGFM_C1)."""

    def __init__(
        self,
        params: HybridParams = None,
        grid: GridParams = None,
    ):
        super().__init__(grid)
        self.p = params or HybridParams()

    # ------------------------------------------------------------------
    @property
    def n_states(self) -> int:
        return 6

    @property
    def state_names(self) -> list:
        return ["delta", "omega", "E", "Pm_slow", "Pm_fast", "Qm"]

    # ------------------------------------------------------------------
    def x0(self, P_ref: float = 0.5, Q_ref: float = 0.0) -> np.ndarray:
        E0 = self.p.E0
        Vg, Xg = self.grid.Vg, self.grid.Xg
        sin_d0 = np.clip(P_ref * Xg / (E0 * Vg), -1.0, 1.0)
        delta0 = float(np.arcsin(sin_d0))
        # Both power filters start at the same operating point
        return np.array([delta0, OMEGA0, E0, P_ref, P_ref, Q_ref], dtype=float)

    # ------------------------------------------------------------------
    def dxdt(
        self,
        t: float,
        x: np.ndarray,
        P_ref: float,
        Q_ref: float,
        omega_grid: float,
    ) -> np.ndarray:
        delta, omega, E, Pm_slow, Pm_fast, Qm = x
        p = self.p

        P_inst, Q_inst = self._grid_power(delta, E)

        # VSM swing equation operates on slowly-measured power
        speed_dev = (omega - OMEGA0) / OMEGA0
        domega = (OMEGA0 / (2.0 * p.H)) * (P_ref - Pm_slow - p.Dp * speed_dev)

        # Feedforward correction: fast droop on pulsating component
        omega_ff = OMEGA0 * p.kff * (P_ref - Pm_fast)
        omega_total = omega + omega_ff

        # Voltage control
        E_ref = p.E0 - p.kq * (Qm - Q_ref)
        dE = (E_ref - E) / p.tau_v

        # Angle integrates total (VSM + FF) frequency
        ddelta = omega_total - omega_grid

        # Power filter dynamics
        dPm_slow = (P_inst - Pm_slow) / p.tau_slow
        dPm_fast = (P_inst - Pm_fast) / p.tau_fast
        dQm = (Q_inst - Qm) / p.tau_q

        return np.array([ddelta, domega, dE, dPm_slow, dPm_fast, dQm])

    # ------------------------------------------------------------------
    def get_outputs(
        self,
        t: float,
        x: np.ndarray,
        P_ref: float,
        Q_ref: float,
        omega_grid: float,
    ) -> dict:
        delta, omega, E, Pm_slow, Pm_fast, Qm = x
        p = self.p
        P_inst, Q_inst = self._grid_power(delta, E)
        omega_ff = OMEGA0 * p.kff * (P_ref - Pm_fast)
        omega_total = omega + omega_ff

        return {
            "delta_deg": float(np.degrees(delta)),
            "P_pu": float(P_inst),
            "Q_pu": float(Q_inst),
            "omega_rads": float(omega_total),
            "freq_hz": float(omega_total / (2.0 * np.pi)),
            "freq_dev_hz": float((omega_total - OMEGA0) / (2.0 * np.pi)),
            "E_pu": float(E),
            "P_pulsating_pu": float(Pm_fast - Pm_slow),
        }
