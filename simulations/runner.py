"""
Simulation engine for GFM inverter models.

Wraps scipy.integrate.solve_ivp (RK45) and packages results into a
SimResult dataclass for convenient post-processing and plotting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Union
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from gfm.models.base import BaseGFM, OMEGA0


Number = Union[int, float]


@dataclass
class SimResult:
    """Container for a single simulation run."""
    model_name: str
    t: np.ndarray                        # time vector  [s]
    X: np.ndarray                        # state matrix  (n_t × n_states)
    state_names: list
    outputs: pd.DataFrame                # physical outputs at each time step
    P_ref_vec: np.ndarray                # P_ref signal used  [pu]
    Q_ref_vec: np.ndarray                # Q_ref signal used  [pu]

    # ------------------------------------------------------------------
    def __repr__(self):
        return (
            f"SimResult(model={self.model_name}, "
            f"t=[{self.t[0]:.3f}, {self.t[-1]:.3f}] s, "
            f"n_steps={len(self.t)})"
        )

    # ------------------------------------------------------------------
    def get(self, key: str) -> np.ndarray:
        """Return an output column by name."""
        return self.outputs[key].to_numpy()

    # ------------------------------------------------------------------
    def freq_nadir_hz(self) -> float:
        return float(self.outputs["freq_hz"].min())

    def freq_peak_hz(self) -> float:
        return float(self.outputs["freq_hz"].max())

    def P_ripple_pu(self) -> float:
        """Peak-to-peak active power ripple."""
        p = self.outputs["P_pu"]
        return float(p.max() - p.min())

    def RoCoF_max(self) -> float:
        """Maximum absolute rate-of-change-of-frequency  [Hz/s]."""
        f = self.outputs["freq_hz"].to_numpy()
        df_dt = np.abs(np.gradient(f, self.t))
        return float(df_dt.max())


# ---------------------------------------------------------------------------
# Core simulation function
# ---------------------------------------------------------------------------

def simulate(
    model: BaseGFM,
    t_span: tuple,
    P_ref: Union[float, Callable[[float], float]] = 0.5,
    Q_ref: Union[float, Callable[[float], float]] = 0.0,
    omega_grid: Union[float, Callable[[float], float]] = OMEGA0,
    x0: np.ndarray = None,
    max_step: float = 1e-3,
    rtol: float = 1e-6,
    atol: float = 1e-8,
    n_eval: int = 2000,
) -> SimResult:
    """
    Integrate a GFM model ODE over t_span.

    Parameters
    ----------
    model       : any BaseGFM subclass instance
    t_span      : (t_start, t_end) in seconds
    P_ref       : float (constant) or callable P_ref(t) → float [pu]
    Q_ref       : float (constant) or callable Q_ref(t) → float [pu]
    omega_grid  : float (constant) or callable ω_grid(t) → float [rad/s]
    x0          : initial state; defaults to model.x0() evaluated at t0 P_ref
    max_step    : solver maximum step size [s]
    rtol, atol  : solver tolerances
    n_eval      : number of evaluation points for output

    Returns
    -------
    SimResult
    """
    # Make all inputs callable
    _P = P_ref if callable(P_ref) else (lambda t, _v=P_ref: _v)
    _Q = Q_ref if callable(Q_ref) else (lambda t, _v=Q_ref: _v)
    _W = omega_grid if callable(omega_grid) else (lambda t, _v=omega_grid: _v)

    # Initial conditions
    if x0 is None:
        P0 = _P(t_span[0])
        Q0 = _Q(t_span[0])
        x0 = model.x0(P0, Q0)

    def rhs(t, x):
        return model.dxdt(t, x, _P(t), _Q(t), _W(t))

    t_eval = np.linspace(t_span[0], t_span[1], n_eval)

    sol = solve_ivp(
        rhs,
        t_span,
        x0,
        method="RK45",
        t_eval=t_eval,
        max_step=max_step,
        rtol=rtol,
        atol=atol,
        dense_output=False,
    )

    if not sol.success:
        raise RuntimeError(f"Solver failed: {sol.message}")

    t_arr = sol.t
    X_arr = sol.y.T  # (n_t, n_states)

    # Build output DataFrame
    rows = []
    for i, ti in enumerate(t_arr):
        xi = X_arr[i]
        out = model.get_outputs(ti, xi, _P(ti), _Q(ti), _W(ti))
        rows.append(out)

    return SimResult(
        model_name=type(model).__name__,
        t=t_arr,
        X=X_arr,
        state_names=model.state_names,
        outputs=pd.DataFrame(rows),
        P_ref_vec=np.array([_P(ti) for ti in t_arr]),
        Q_ref_vec=np.array([_Q(ti) for ti in t_arr]),
    )
