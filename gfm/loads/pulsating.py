"""
Pulsating load profiles representative of AI / data-centre workloads.

Frequency range of interest: 0.1 – 30 Hz (per REGFM_C1 design target).

Usage
-----
    load = PulsatingLoad(
        P_base=0.5,
        components=[(5.0, 0.15), (15.0, 0.08)],   # (Hz, amplitude_pu)
    )
    P_ref = load(t)   # callable, returns P_ref at time t
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Tuple
import numpy as np


@dataclass
class LoadProfile:
    """Single sinusoidal pulsation component."""
    freq_hz: float          # oscillation frequency [Hz]
    amplitude: float        # peak amplitude [pu]
    phase_rad: float = 0.0  # initial phase [rad]


@dataclass
class PulsatingLoad:
    """
    Superposition of sinusoidal pulsation components on a base load.

    Parameters
    ----------
    P_base      : steady-state active power [pu]
    components  : list of (freq_hz, amplitude_pu) tuples or LoadProfile objects
    Q_base      : steady-state reactive power [pu]
    t_start     : time at which pulsation begins [s]
    """

    P_base: float = 0.5
    components: List = field(default_factory=list)
    Q_base: float = 0.0
    t_start: float = 0.0

    def __post_init__(self):
        parsed = []
        for c in self.components:
            if isinstance(c, LoadProfile):
                parsed.append(c)
            else:
                freq, amp = c[0], c[1]
                phase = c[2] if len(c) > 2 else 0.0
                parsed.append(LoadProfile(freq, amp, phase))
        self.components = parsed

    # ------------------------------------------------------------------
    def P_ref(self, t: float) -> float:
        """Active power reference at time t [pu]."""
        if t < self.t_start:
            return self.P_base
        P = self.P_base
        for c in self.components:
            P += c.amplitude * np.sin(2.0 * np.pi * c.freq_hz * t + c.phase_rad)
        return float(P)

    def Q_ref(self, t: float) -> float:
        """Reactive power reference at time t [pu] (constant by default)."""
        return float(self.Q_base)

    def __call__(self, t: float) -> float:
        """Shorthand for P_ref(t)."""
        return self.P_ref(t)

    # ------------------------------------------------------------------
    @staticmethod
    def step(P_before: float, P_after: float, t_step: float = 1.0) -> "PulsatingLoad":
        """Factory: simple load step (no oscillation)."""
        class _Step:
            def P_ref(self, t):
                return P_after if t >= t_step else P_before

            def Q_ref(self, t):
                return 0.0

            def __call__(self, t):
                return self.P_ref(t)

        return _Step()

    # ------------------------------------------------------------------
    @staticmethod
    def ai_datacenter(
        P_base: float = 0.6,
        t_start: float = 0.5,
    ) -> "PulsatingLoad":
        """
        Representative AI / data-centre multi-frequency load profile.

        Components (approximate field observations):
          0.3 Hz  – cooling system oscillation
          2.0 Hz  – server rack power cycling
          8.0 Hz  – GPU batch inference duty cycle
          22.0 Hz – switched-mode PSU inter-harmonic
        """
        return PulsatingLoad(
            P_base=P_base,
            t_start=t_start,
            components=[
                LoadProfile(0.3,  0.06),
                LoadProfile(2.0,  0.10),
                LoadProfile(8.0,  0.07, phase_rad=np.pi / 4),
                LoadProfile(22.0, 0.04, phase_rad=np.pi / 6),
            ],
        )
