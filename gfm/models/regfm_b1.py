"""
REGFM_B1 – Virtual Synchronous Machine (VSM) Control
=====================================================

Control law
-----------
Swing equation (virtual inertia):
    dω/dt = (ω₀ / 2H) · [P_ref − Pm − Dp · (ω − ω₀)/ω₀]

Voltage control (Q–V droop):
    dE/dt = (1/τᵥ) · [E₀ − kq·(Qm − Q_ref) − E]

State vector  x = [δ, ω, E, Pm, Qm]
    δ  : power angle  [rad]
    ω  : virtual rotor angular frequency  [rad/s]
    E  : internal voltage magnitude  [pu]
    Pm : low-pass-filtered active power  [pu]
    Qm : low-pass-filtered reactive power  [pu]

Characteristic behaviour
------------------------
* Synthetic inertia via the swing equation – resists rapid frequency changes.
* Frequency nadir is improved compared with droop for large load steps.
* Inertia can amplify mid-frequency oscillations (1–10 Hz range) because
  the virtual rotor "stores" energy and returns it with a phase lag.
* This is why pulsating AI/data-centre loads (0.1–30 Hz) can excite
  oscillatory modes that do NOT appear in droop or hybrid control.
"""

from dataclasses import dataclass
import numpy as np

from .base import BaseGFM, GridParams, OMEGA0


@dataclass
class VSMParams:
    H: float = 5.0        # virtual inertia constant  [s]  (equivalent to sync. gen.)
    Dp: float = 10.0      # per-unit damping coefficient  [pu/pu]
    kq: float = 0.05      # Q–V droop gain  [pu/pu]
    tau_p: float = 0.05   # active-power filter time constant  [s]
    tau_q: float = 0.05   # reactive-power filter time constant  [s]
    tau_v: float = 0.05   # voltage control time constant  [s]
    E0: float = 1.0       # voltage setpoint  [pu]


class REGFM_B1(BaseGFM):
    """Virtual Synchronous Machine GFM inverter (REGFM_B1)."""

    def __init__(
        self,
        params: VSMParams = None,
        grid: GridParams = None,
    ):
        super().__init__(grid)
        self.p = params or VSMParams()

    # ------------------------------------------------------------------
    @property
    def n_states(self) -> int:
        return 5

    @property
    def state_names(self) -> list:
        return ["delta", "omega", "E", "Pm", "Qm"]

    # ------------------------------------------------------------------
    def x0(self, P_ref: float = 0.5, Q_ref: float = 0.0) -> np.ndarray:
        E0 = self.p.E0
        Vg, Xg = self.grid.Vg, self.grid.Xg
        sin_d0 = np.clip(P_ref * Xg / (E0 * Vg), -1.0, 1.0)
        delta0 = float(np.arcsin(sin_d0))
        return np.array([delta0, OMEGA0, E0, P_ref, Q_ref], dtype=float)

    # ------------------------------------------------------------------
    def dxdt(
        self,
        t: float,
        x: np.ndarray,
        P_ref: float,
        Q_ref: float,
        omega_grid: float,
    ) -> np.ndarray:
        delta, omega, E, Pm, Qm = x
        p = self.p

        P_inst, Q_inst = self._grid_power(delta, E)

        # Swing equation
        speed_dev = (omega - OMEGA0) / OMEGA0   # per-unit speed deviation
        domega = (OMEGA0 / (2.0 * p.H)) * (P_ref - Pm - p.Dp * speed_dev)

        # Voltage control
        E_ref = p.E0 - p.kq * (Qm - Q_ref)
        dE = (E_ref - E) / p.tau_v

        ddelta = omega - omega_grid
        dPm = (P_inst - Pm) / p.tau_p
        dQm = (Q_inst - Qm) / p.tau_q

        return np.array([ddelta, domega, dE, dPm, dQm])

    # ------------------------------------------------------------------
    def get_outputs(
        self,
        t: float,
        x: np.ndarray,
        P_ref: float,
        Q_ref: float,
        omega_grid: float,
    ) -> dict:
        delta, omega, E, Pm, Qm = x
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
