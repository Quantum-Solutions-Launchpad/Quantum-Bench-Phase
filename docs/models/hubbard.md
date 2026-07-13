# Hubbard

The **Hubbard model** lives on the same honeycomb lattice with two sublattices $A$ and $B$, but adds the two ingredients the non-interacting models lack: an electron spin $\sigma \in \{\uparrow,\downarrow\}$ and an onsite Coulomb repulsion $U$:

$$
H = -t \sum_{\langle i,j\rangle,\sigma} c_{i,\sigma}^\dagger c_{j,\sigma}
    + U \sum_i c_{i,\uparrow}^\dagger c_{i,\uparrow}\, c_{i,\downarrow}^\dagger c_{i,\downarrow},
$$

where $t$ is the nearest-neighbor hopping and $U$ penalizes double occupancy. The interaction term means the energy can no longer be obtained from a single-particle Hamiltonian—the full many-body Hamiltonian, exponential in the number of sites, must be used instead. That exponential cost is exactly what makes the Hubbard model a compelling target for quantum simulation (see [Concepts](../getting-started/concepts.md)).

## Parameters

| Parameter | Symbol | Meaning |
| --- | --- | --- |
| `t` | $t$ | Nearest-neighbor hopping ($A \leftrightarrow B$) |
| `U` | $U$ | Onsite Coulomb repulsion (double-occupancy penalty) |

## Typical Sweep

The Hubbard model's magnetic phases are the object of interest: varying $U$, $t$, and the filling $N_\text{occ}$ moves the ground state between paramagnetic, ferromagnetic, and antiferromagnetic order. These are read off two magnetization observables—the staggered magnetization `M_stag` ($M_A - M_B$) and the total magnetization `M_total` ($M_A + M_B$), both available alongside the usual `E` and `gap`. The canonical sweep is $N_\text{occ}$ against $U$ at fixed $t$, which resolves how magnetic order onsets with interaction strength near half-filling.

## Example

The following sweeps the filling `n_occ` against $U$ on a $2\times 2$ lattice and reads off the staggered magnetization, tracing the magnetic phase diagram as a heatmap. The underlying specification lives in `qbp/models/hubbard.yaml`.

```{jupyter-execute}
:hide-code:

import io
import sys
from pathlib import Path
from loguru import logger

logger.remove()
sys.stdout = sys.stderr = io.StringIO()

import qbp
from qbp import Method


def _find_data_dir() -> Path:
    for base in (Path.cwd(), *Path.cwd().parents):
        for candidate in (base / "docs" / "_data", base / "_data"):
            if candidate.is_dir():
                return candidate
    raise FileNotFoundError("docs/_data not found relative to cwd")


_DATA_DIR = _find_data_dir()


def _patched_run(*args, **kwargs):
    result = qbp.load_result(str(_DATA_DIR / "hubbard-2x2-M_stag-n_occ-vs-U.json"))
    result.plot(hide_plot=kwargs.get("hide_plot", False))
    return result


qbp.run = _patched_run
```

```{jupyter-execute}
import qbp
from qbp import Method

result = qbp.run(
    model="hubbard",
    method=[Method.ANALYTIC],
    lattice=(2, 2),
    x_param="n_occ",
    y_param="U",
    y_range=(0.0, 10.0, 0.5),
    model_params={"t": 1.0},
    observable="M_stag",
    heatmap=True,
)
```

The staggered magnetization stays near zero at weak coupling and small filling, then swells into an antiferromagnetic lobe around half-filling ($N_\text{occ} = 8$) as $U$ grows—the interaction-driven magnetic order that makes the Hubbard model interesting.

```{note}
Analytic runs on an interacting model perform full many-body exact diagonalization, whose cost grows *exponentially* with the number of sites, so even the $2\times 2$ lattice above is already a sizeable computation. To push the magnetic phase diagram to larger lattices, drive the same sweep with simulation (`Method.VQE`, `Method.IQPE`, `Method.DMRG`; see [Performing Simulation](../user-guide/performing-simulation.md)).
```
