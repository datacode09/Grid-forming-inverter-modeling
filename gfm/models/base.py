from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

OMEGA0: float = 2.0 * np.pi * 60.0   # 376.991 rad/s  (US 60 Hz base)


@dataclass
class GridParams:
    """Thevenin-equivalent infinite-bus grid parameters (per-unit)."""
    Xg: float = 0.3    # grid reactance [pu]
    Vg: float = 1.0    # infinite-bus voltage magnitude [pu]


class BaseGFM(ABC):
    """
    Abstract base class for Grid-Forming (GFM) inverter control models.

    Coordinate convention
    ---------------------
    All models operate in the synchronous reference frame aligned with the
    infinite-bus voltage (θ_grid = 0).  The power angle δ is the angle of
    the inverter internal voltage relative to the grid bus.

    Power-flow equations (lossless line approximation):
        P = (E · Vg / Xg) · sin(δ)
        Q = (E² − E · Vg · cos(δ)) / Xg
    """

    def __init__(self, grid: GridParams = None):
        self.grid = grid or GridParams()

    # ------------------------------------------------------------------
    # Interface that every subclass must implement
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def n_states(self) -> int:
        """Number of ODE states."""
        ...

    @property
    @abstractmethod
    def state_names(self) -> list:
        """Human-readable name for each state variable."""
        ...

    @abstractmethod
    def x0(self, P_ref: float = 0.5, Q_ref: float = 0.0) -> np.ndarray:
        """Return steady-state initial condition vector."""
        ...

    @abstractmethod
    def dxdt(
        self,
        t: float,
        x: np.ndarray,
        P_ref: float,
        Q_ref: float,
        omega_grid: float,
    ) -> np.ndarray:
        """
        Compute time derivatives of the state vector.

        Parameters
        ----------
        t          : current simulation time [s]
        x          : state vector
        P_ref      : active power reference at time t [pu]
        Q_ref      : reactive power reference at time t [pu]
        omega_grid : grid angular frequency at time t [rad/s]

        Returns
        -------
        dx/dt  numpy array of same shape as x
        """
        ...

    @abstractmethod
    def get_outputs(
        self,
        t: float,
        x: np.ndarray,
        P_ref: float,
        Q_ref: float,
        omega_grid: float,
    ) -> dict:
        """Return dictionary of physical output quantities."""
        ...

    # ------------------------------------------------------------------
    # Shared helper
    # ------------------------------------------------------------------

    def _grid_power(self, delta: float, E: float):
        """Active and reactive power injected toward the infinite bus."""
        Xg, Vg = self.grid.Xg, self.grid.Vg
        P = (E * Vg / Xg) * np.sin(delta)
        Q = (E ** 2 - E * Vg * np.cos(delta)) / Xg
        return P, Q
