"""
Reference-frame transformation utilities.

Conventions
-----------
* abc → αβ0  : Clarke (amplitude-invariant)
* αβ0 → dq0  : Park (rotation by angle θ)
* abc → dq0  : combined Park transform

All angles in radians.  Positive-sequence a-phase leads b and c by 120°.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Clarke transform  (abc → αβ0, amplitude-invariant)
# ---------------------------------------------------------------------------

_C = (2.0 / 3.0) * np.array(
    [
        [1.0, -0.5, -0.5],
        [0.0, np.sqrt(3) / 2.0, -np.sqrt(3) / 2.0],
        [0.5, 0.5, 0.5],
    ]
)
_C_inv = np.linalg.inv(_C)


def clarke_transform(v_abc: np.ndarray) -> np.ndarray:
    """abc → [α, β, 0] (amplitude-invariant Clarke)."""
    return _C @ np.asarray(v_abc)


def clarke_inverse(v_ab0: np.ndarray) -> np.ndarray:
    """[α, β, 0] → abc."""
    return _C_inv @ np.asarray(v_ab0)


# ---------------------------------------------------------------------------
# Park transform  (αβ0 → dq0)
# ---------------------------------------------------------------------------

def park_transform(v_ab0: np.ndarray, theta: float) -> np.ndarray:
    """
    Rotate αβ frame to dq frame by angle theta [rad].

    [d]   [ cos θ   sin θ  0 ] [α]
    [q] = [-sin θ   cos θ  0 ] [β]
    [0]   [  0       0     1 ] [0]
    """
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    return R @ np.asarray(v_ab0)


def park_inverse(v_dq0: np.ndarray, theta: float) -> np.ndarray:
    """dq0 → αβ0."""
    c, s = np.cos(theta), np.sin(theta)
    R_inv = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return R_inv @ np.asarray(v_dq0)


# ---------------------------------------------------------------------------
# Combined abc ↔ dq0
# ---------------------------------------------------------------------------

def abc_to_dq0(v_abc: np.ndarray, theta: float) -> np.ndarray:
    """abc → dq0 via Clarke then Park."""
    return park_transform(clarke_transform(v_abc), theta)


def dq0_to_abc(v_dq0: np.ndarray, theta: float) -> np.ndarray:
    """dq0 → abc via inverse Park then inverse Clarke."""
    return clarke_inverse(park_inverse(v_dq0, theta))


# ---------------------------------------------------------------------------
# Convenience: per-unit power from dq voltages & currents
# ---------------------------------------------------------------------------

def power_dq(v_dq: np.ndarray, i_dq: np.ndarray) -> tuple:
    """
    Instantaneous active and reactive power in dq frame (per-unit, 3-phase).

    P = 3/2 · (vd·id + vq·iq)
    Q = 3/2 · (vq·id − vd·iq)
    """
    vd, vq = v_dq[0], v_dq[1]
    id_, iq = i_dq[0], i_dq[1]
    P = 1.5 * (vd * id_ + vq * iq)
    Q = 1.5 * (vq * id_ - vd * iq)
    return P, Q
