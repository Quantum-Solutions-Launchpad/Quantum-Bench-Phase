# Haldane

The **Haldane model** is a spinless tight-binding model on a honeycomb lattice with two sublattices, $A$ and $B$. Electrons hop between nearest-neighbor sites with amplitude $t_1$ and between next-nearest-neighbor sites with a complex amplitude $t_2 e^{\pm i\phi}$, whose sign depends on the orientation of the hop. A staggered onsite potential $\pm M$ distinguishes the two sublattices:

$$
H = -t_1 \sum_{\langle i,j \rangle} c_i^\dagger c_j
    -t_2 \sum_{\langle\langle i,j \rangle\rangle} e^{i \nu_{ij} \phi}\, c_i^\dagger c_j
    + M \sum_i \xi_i\, c_i^\dagger c_i,
$$

where $\xi_i = +1$ on sublattice $A$ and $-1$ on $B$, and $\nu_{ij} = \pm 1$ encodes the chirality of the next-nearest-neighbor hop. The competition between the staggered mass $M$ and the time-reversal-breaking hopping $t_2$ drives a topological phase transition between a trivial insulator and quantum-Hall phases with Chern number $\nu = \pm 1$. Because it is non-interacting, the Haldane model is efficiently solvable by exact diagonalization; due to that and its interesting physical properties and phase diagrams, we use it as a canonical reference for our library.

## Parameters

| Parameter | Symbol | Meaning |
| --- | --- | --- |
| `t1`  | $t_1$  | Nearest-neighbor hopping ($A \leftrightarrow B$) |
| `t2`  | $t_2$  | Next-nearest-neighbor hopping magnitude ($A \to A$, $B \to B$) |
| `phi` | $\phi$ | Haldane flux phase on the next-nearest-neighbor hop |
| `M`   | $M$    | Staggered sublattice mass ($+M$ on $A$, $-M$ on $B$) |

## Typical Sweep

The canonical phase diagram plots $M$ against $\phi$ at a fixed $t_2$. The zero-field quantum-Hall phases occupy the region $|M/t_2| < 3\sqrt{3}\,\left|\sin\phi\right|$, so the phase boundary appears as two sinusoidal lobes symmetric about $\phi = 0$, with peaks at $\pm 3\sqrt{3}\,t_2$. Reading the spectral gap instead of the ground-state energy renders that boundary sharply as a zero-gap curve.

## Example

The following sweeps $\phi$ against $M$ at $t_2 = 0.1$ on a $3\times 3$ honeycomb lattice and reads off the spectral gap. The underlying specification lives in `qbp/models/haldane.yaml`.

```{jupyter-execute}
:hide-code:

import io
import sys
from loguru import logger

logger.remove()
sys.stdout = sys.stderr = io.StringIO()
```

```{jupyter-execute}
import math
import qbp
from qbp import Method

result = qbp.run(
    model="haldane",
    method=[Method.ANALYTIC],
    lattice=(3, 3),
    x_param="phi",
    x_range=(-math.pi, math.pi, math.pi / 100),
    y_param="M",
    y_range=(-1.0, 1.0, 0.01),
    model_params={"t1": 1.0, "t2": 0.1},
    observable="gap",
    heatmap=True,
)
```

The gap closes along the two sinusoidal lobes peaking at $\pm 3\sqrt{3} \cdot 0.1 \approx 0.5$, marking the boundary between the trivial insulator and the two topological phases. The Haldane model also implements a Bloch Hamiltonian, so it additionally supports momentum-space [band-structure](../more-examples/band-structure.md) runs.
