"""
REGFM_A1 – Droop-based Grid-Forming Control
============================================

Control law
-----------
    ω  = ω₀ · [1 − kp · (Pm − P_ref)]      (P–ω droop)
    E  = E₀ − kq · (Qm − Q_ref)             (Q–V droop)

State vector  x = [δ, Pm, Qm]
    δ  : power angle  [rad]
    Pm : low-pass-filtered active power  [pu]
    Qm : low-pass-filtered reactive power  [pu]

Characteristic behaviour
------------------------
* Pure algebraic frequency response – no synthetic inertia.
* Frequency settles instantly to the droop characteristic.
* Good steady-state load sharing; poor dynamic rejection of fast
  pulsating loads (limited by τp filter bandwidth ~1/τp Hz).
"""

from dataclasses import dataclass
import numpy as np

from .base import BaseGFM, GridParams, OMEGA0


@dataclass
class DroopParams:
    kp: float = 0.05      # P–ω droop gain  [pu/pu]  (5 % droop)
    kq: float = 0.05      # Q–V droop gain  [pu/pu]  (5 % droop)
    tau_p: float = 0.05   # active-power filter time constant  [s]
    tau_q: float = 0.05   # reactive-power filter time constant  [s]
    E0: float = 1.0       # no-load voltage setpoint  [pu]


class REGFM_A1(BaseGFM):
    """Droop-based GFM inverter (REGFM_A1)."""

    def __init__(
        self,
        params: DroopParams = None,
        grid: GridParams = None,
    ):
        super().__init__(grid)
        self.p = params or DroopParams()

    # ------------------------------------------------------------------
    @property
    def n_states(self) -> int:
        return 3

    @property
    def state_names(self) -> list:
        return ["delta", "Pm", "Qm"]

    # ------------------------------------------------------------------
    def x0(self, P_ref: float = 0.5, Q_ref: float = 0.0) -> np.ndarray:
        """Steady-state initial conditions for the given operating point."""
        E0 = self.p.E0
        Vg, Xg = self.grid.Vg, self.grid.Xg
        # δ from  P = (E·Vg/Xg)·sin(δ)
        sin_d0 = np.clip(P_ref * Xg / (E0 * Vg), -1.0, 1.0)
        delta0 = float(np.arcsin(sin_d0))
        return np.array([delta0, P_ref, Q_ref], dtype=float)

    # ------------------------------------------------------------------
    def dxdt(
        self,
        t: float,
        x: np.ndarray,
        P_ref: float,
        Q_ref: float,
        omega_grid: float,
    ) -> np.ndarray:
        delta, Pm, Qm = x
        p = self.p

        E = p.E0 - p.kq * (Qm - Q_ref)
        omega = OMEGA0 * (1.0 - p.kp * (Pm - P_ref))

        P_inst, Q_inst = self._grid_power(delta, E)

        ddelta = omega - omega_grid
        dPm = (P_inst - Pm) / p.tau_p
        dQm = (Q_inst - Qm) / p.tau_q

        return np.array([ddelta, dPm, dQm])

    # ------------------------------------------------------------------
    def get_outputs(
        self,
        t: float,
        x: np.ndarray,
        P_ref: float,
        Q_ref: float,
        omega_grid: float,
    ) -> dict:
        delta, Pm, Qm = x
        p = self.p
        E = p.E0 - p.kq * (Qm - Q_ref)
        omega = OMEGA0 * (1.0 - p.kp * (Pm - P_ref))
        P_inst, Q_inst = self._grid_power(delta, E)

        return {
            "delta_deg": float(np.degrees(delta)),
            "P_pu": float(P_inst),
            "Q_pu": float(Q_inst),
            "omega_rads": float(omega),
            "freq_hz": float(omega / (2.0 * np.pi)),
            "freq_dev_hz": float((omega - OMEGA0) / (2.0 * np.pi)),
            "E_pu": float(E),
        }
