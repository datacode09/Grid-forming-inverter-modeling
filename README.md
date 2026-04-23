# Grid-Forming Inverter Modeling

Python simulation library for three generic Grid-Forming (GFM) inverter
control architectures, with emphasis on the **REGFM_C1 hybrid** topology
designed to handle pulsating AI / data-centre loads (0.1–30 Hz).

---

## Control architectures

| Identifier | Control strategy | Key feature |
|---|---|---|
| **REGFM_A1** | Droop-based | Algebraic P–ω and Q–V droop |
| **REGFM_B1** | Virtual Synchronous Machine (VSM) | Swing-equation virtual inertia |
| **REGFM_C1** | Hybrid GFM | VSM base + high-bandwidth feedforward path |

### Why REGFM_C1?

Droop and VSM both react to power deviations through relatively slow
feedback loops.  For pulsating loads in the **0.1–30 Hz** band — typical
of AI inference clusters, GPU racks, and high-density data centres — the
slow loops cannot compensate fast enough:

- **A1**: no inertia, frequency jumps instantly but power recovery is
  bandwidth-limited by the measurement filter.
- **B1**: virtual inertia smooths RoCoF but the stored "virtual kinetic
  energy" can amplify mid-frequency oscillations.
- **C1**: a **feedforward path** on the power/current control loop directly
  measures and cancels pulsating components without waiting for the slow
  VSM dynamics.  Result: dramatically lower frequency deviation and power
  ripple.

---

## Repository layout

```
gfm/
  models/
    base.py         BaseGFM abstract class + GridParams
    regfm_a1.py     Droop-based GFM (REGFM_A1)
    regfm_b1.py     Virtual Synchronous Machine (REGFM_B1)
    regfm_c1.py     Hybrid GFM with feedforward (REGFM_C1)
  loads/
    pulsating.py    PulsatingLoad – multi-frequency load profiles
  utils/
    transforms.py   Clarke / Park / dq0 reference-frame transforms
simulations/
  runner.py         simulate() wrapper around scipy solve_ivp
examples/
  01_droop_load_step.py     A1 load-step response
  02_vsm_inertia.py         A1 vs B1 synthetic-inertia comparison
  03_hybrid_pulsating.py    A1 / B1 / C1 vs AI data-centre load
  04_compare_all.py         Side-by-side across three scenarios
```

---

## Quick start

```bash
pip install -r requirements.txt
cd examples
python 03_hybrid_pulsating.py
python 04_compare_all.py
```

---

## Physical model

All three inverters use the same **simplified lossless power-flow** equations
for a single inverter connected to an infinite bus through reactance *X*:

```
P = (E · Vg / Xg) · sin(δ)
Q = (E² − E · Vg · cos(δ)) / Xg
```

where `δ` is the power angle (inverter internal voltage vs. grid bus),
`E` is the inverter output voltage magnitude, and `Vg = 1 pu` is the
infinite-bus voltage.

All quantities are in **per-unit** (MVA, kV base).  Frequency base: 60 Hz.

### REGFM_C1 feedforward detail

```
ω_vsm   : VSM virtual-rotor frequency (slow, τ_slow ≈ 0.1 s)
ω_ff    : kff · ω₀ · (P_ref − Pm_fast)   (τ_fast ≈ 0.005 s → ~32 Hz BW)
ω_total = ω_vsm + ω_ff
δ̇       = ω_total − ω_grid
```

The `ω_ff` term acts as a high-bandwidth proportional droop that compensates
pulsating power without exciting the slow VSM inertia dynamics.

---

## Key parameters

### Grid
| Parameter | Default | Description |
|---|---|---|
| `Xg` | 0.3 pu | Grid reactance |
| `Vg` | 1.0 pu | Infinite-bus voltage |

### REGFM_A1
| Parameter | Default | Description |
|---|---|---|
| `kp` | 0.05 | P–ω droop gain (5 % droop) |
| `kq` | 0.05 | Q–V droop gain |
| `tau_p` | 0.05 s | Active power filter |

### REGFM_B1
| Parameter | Default | Description |
|---|---|---|
| `H` | 5.0 s | Virtual inertia constant |
| `Dp` | 10.0 | Per-unit damping |
| `kq` | 0.05 | Q–V droop gain |

### REGFM_C1
| Parameter | Default | Description |
|---|---|---|
| `H` | 5.0 s | VSM inertia constant |
| `Dp` | 10.0 | VSM damping |
| `kff` | 0.10 | Feedforward gain |
| `tau_fast` | 0.005 s | Fast power filter (~32 Hz BW) |
| `tau_slow` | 0.10 s | Slow power filter (~1.6 Hz BW) |

---

## Extending the library

Subclass `BaseGFM` and implement four methods:

```python
from gfm.models.base import BaseGFM, GridParams
import numpy as np

class MyGFM(BaseGFM):
    @property
    def n_states(self): return 3

    @property
    def state_names(self): return ["delta", "Pm", "Qm"]

    def x0(self, P_ref=0.5, Q_ref=0.0):
        ...

    def dxdt(self, t, x, P_ref, Q_ref, omega_grid):
        ...

    def get_outputs(self, t, x, P_ref, Q_ref, omega_grid):
        ...
```

Then pass it directly to `simulate()`.

---

## References

- IEEE Std 2800-2022 — *Interconnection and Interoperability of Inverter-Based Resources*
- Rocabert et al., "Control of Power Converters in AC Microgrids," *IEEE Trans. Power Electron.*, 2012
- D'Arco & Suul, "Virtual Synchronous Machines — Classification of Implementations and Analysis of Equivalence to Droop Controllers," *IEEE Grenoble PowerTech*, 2013
- NERC, "Grid Forming Technology in Energy Storage Systems," 2021
