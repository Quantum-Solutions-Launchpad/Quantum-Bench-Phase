# Hubbard

The **Hubbard model** adds the two ingredients the non-interacting models lack: an electron spin $\sigma \in \{\uparrow,\downarrow\}$ and an onsite Coulomb repulsion $U$:

$$
H = -t \sum_{\langle i,j\rangle,\sigma} c_{i,\sigma}^\dagger c_{j,\sigma}
    + U \sum_i c_{i,\uparrow}^\dagger c_{i,\uparrow}\, c_{i,\downarrow}^\dagger c_{i,\downarrow},
$$

where $t$ is the nearest-neighbor hopping and $U$ penalizes double occupancy. The interaction term means the energy can no longer be obtained from a single-particle Hamiltonian—the full many-body Hamiltonian, exponential in the number of sites, must be used instead. That exponential cost is exactly what makes the Hubbard model a compelling target for quantum simulation (see [Concepts](../getting-started/concepts.md)).

## Lattices

The Hubbard model ships on three lattices. The Hamiltonian above is the same in each; only the bond graph changes, and with it the magnetic physics:

| Registry name | Lattice | Coordination | Bipartite | Non-interacting bandwidth |
| --- | --- | --- | --- | --- |
| `hubbard-honeycomb`  | honeycomb  | 3 | yes | $[-3t, 3t]$ |
| `hubbard-square`     | square     | 4 | yes | $[-4t, 4t]$ |
| `hubbard-triangular` | triangular | 6 | no  | $[-6t, 3t]$ |

The honeycomb and square lattices are bipartite: the two sublattices $A$ and $B$ of the unit cell are the two sublattices of the Néel pattern, the spectrum is particle–hole symmetric, and the staggered structure factor `S_stag` is a genuine antiferromagnetic order parameter that grows with $U$ at half filling.

The triangular lattice is **frustrated**. Its unit cell has a single site, so there is no sublattice to stagger—`S_stag` and `S_total` return the same value there—its non-interacting band is asymmetric, and no Néel pattern can satisfy every bond. Neither structure factor grows with $U$: the ordering the triangular lattice does favor is the 120° spiral at the Brillouin-zone corner, which is not the wavevector either observable measures. That frustration is what makes the triangular Hubbard model the standard starting point for quantum-spin-liquid physics.

```{note}
`lattice=(Lx, Ly)` counts unit cells. A $2\times2$ honeycomb or square lattice is 8 sites (16 qubits), while a $2\times2$ triangular lattice is 4 sites (8 qubits).
```

## Parameters

| Parameter | Symbol | Meaning |
| --- | --- | --- |
| `t` | $t$ | Nearest-neighbor hopping ($A \leftrightarrow B$) |
| `U` | $U$ | Onsite Coulomb repulsion (double-occupancy penalty) |

## Typical Sweep

The Hubbard model's magnetic phases are the object of interest: varying $U$, $t$, and the filling $N_\text{occ}$ moves the ground state between paramagnetic, ferromagnetic, and antiferromagnetic order. These are read off two spin structure factors—the staggered `S_stag` and the total `S_total`, both available alongside the usual `E` and `gap`. On the bipartite honeycomb and square lattices `S_stag` is the antiferromagnetic order parameter and grows with $U$ near half filling; on the frustrated triangular lattice it degenerates into `S_total` and stays flat, which is the signature of the frustration rather than a bug. The canonical sweep is $N_\text{occ}$ against $U$ at fixed $t$, which resolves how magnetic order onsets with interaction strength near half-filling.

## Example

The following sweeps the filling `n_occ` against $U$ on a $2\times 2$ lattice and reads off the staggered spin structure factor, tracing the magnetic phase diagram as a heatmap. The underlying specification lives in `qbp/models/hubbard-honeycomb.yaml`; swapping in `"hubbard-square"` or `"hubbard-triangular"` runs the identical sweep on the other lattices.

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
    model="hubbard-honeycomb",
    method=[Method.ANALYTIC],
    lattice=(2, 2),
    x_param="n_occ",
    y_param="U",
    y_range=(0.0, 10.0, 0.5),
    model_params={"t": 1.0},
    observable="S_stag",
    heatmap=True,
)
```

The staggered structure factor stays near zero at weak coupling and small filling, then swells into an antiferromagnetic lobe around half-filling ($N_\text{occ} = 8$) as $U$ grows—the interaction-driven magnetic order that makes the Hubbard model interesting.

```{note}
Analytic runs on an interacting model perform full many-body exact diagonalization, whose cost grows *exponentially* with the number of sites, so even the $2\times 2$ lattice above is already a sizeable computation. To push the magnetic phase diagram to larger lattices, drive the same sweep with simulation (`Method.VQE`, `Method.IQPE`, `Method.DMRG`; see [Performing Simulation](../user-guide/performing-simulation.md)).
```
