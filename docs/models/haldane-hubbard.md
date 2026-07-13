# Haldane–Hubbard

The **Haldane–Hubbard model** combines the topological band structure of the [Haldane model](haldane.md) with the onsite interaction of the [Hubbard model](hubbard.md). On the honeycomb lattice it keeps the staggered mass $M$ and the complex next-nearest-neighbor hopping $t_2 e^{\pm i\phi}$ that break time-reversal symmetry, and adds a spin and an onsite Coulomb repulsion $U$:

$$
H = -t_1 \sum_{\langle i,j\rangle,\sigma} c_{i,\sigma}^\dagger c_{j,\sigma}
    -t_2 \sum_{\langle\langle i,j\rangle\rangle,\sigma} e^{i\nu_{ij}\phi}\, c_{i,\sigma}^\dagger c_{j,\sigma}
    + M \sum_i \xi_i\, c_i^\dagger c_i
    + U \sum_i c_{i,\uparrow}^\dagger c_{i,\uparrow}\, c_{i,\downarrow}^\dagger c_{i,\downarrow}.
$$

The interplay of topology and interactions makes it a richer target than either parent: the interaction $U$ competes with the topological gap set by $t_2$ and $\phi$, and can drive magnetic order on top of the underlying band topology. Like the Hubbard model, it is genuinely interacting, so its ground-state energy requires the full many-body Hamiltonian.

## Parameters

| Parameter | Symbol | Meaning |
| --- | --- | --- |
| `t1`  | $t_1$  | Nearest-neighbor hopping ($A \leftrightarrow B$) |
| `t2`  | $t_2$  | Next-nearest-neighbor hopping magnitude ($A \to A$, $B \to B$) |
| `phi` | $\phi$ | Haldane flux phase on the next-nearest-neighbor hop |
| `M`   | $M$    | Staggered sublattice mass ($+M$ on $A$, $-M$ on $B$) |
| `U`   | $U$    | Onsite Coulomb repulsion (double-occupancy penalty) |

## Typical Sweep

As with the Hubbard model, the interesting axes are interaction versus filling. The canonical sweep is $N_\text{occ}$ against $U$ at fixed hopping, flux, and mass, tracking the onset of magnetic order (read off the staggered magnetization `M_stag` or total magnetization `M_total`) as the repulsion grows. Sweeping $t_2$ or $\phi$ against $U$ instead probes how interactions reshape the topological gap.

## Example

The following sweeps the filling `n_occ` against $U$ on a $2\times 2$ lattice at a fixed topological point ($t_2 = 0.2$, $\phi = \pi/4$, $M = 0.1$) and reads off the ground-state energy, giving the interacting energy landscape across filling and interaction strength. The underlying specification lives in `qbp/models/haldane_hubbard.yaml`.

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
    result = qbp.load_result(str(_DATA_DIR / "haldane-hubbard-2x2-n_occ-vs-U.json"))
    result.plot(hide_plot=kwargs.get("hide_plot", False))
    return result


qbp.run = _patched_run
```

```{jupyter-execute}
import math
import qbp
from qbp import Method

result = qbp.run(
    model="haldane-hubbard",
    method=[Method.ANALYTIC],
    lattice=(2, 2),
    x_param="n_occ",
    y_param="U",
    y_range=(0.0, 4.0, 1.0),
    model_params={"t1": 1.0, "t2": 0.2, "phi": math.pi / 4, "M": 0.1},
)
```

The energy dips as the lower band fills toward half-filling and rises with $U$ as double occupancy is penalized. Swap `observable="M_stag"` (or `"M_total"`) to read the magnetization instead, or sweep $t_2$ or $\phi$ against $U$ to probe how interactions reshape the topological gap.

```{note}
Analytic runs on an interacting model perform full many-body exact diagonalization, whose cost grows *exponentially* with the number of sites, so even the $2\times 2$ lattice above is already a sizeable computation. To push the magnetic phase diagram to larger lattices, drive the same sweep with simulation (`Method.VQE`, `Method.IQPE`, `Method.DMRG`; see [Performing Simulation](../user-guide/performing-simulation.md)).
```
