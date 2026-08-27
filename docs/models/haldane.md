# Haldane

The **Haldane model** is a spinless tight-binding model on a two-sublattice lattice ($A$ and $B$). Electrons hop between nearest-neighbor sites with amplitude $t_1$ and pick up a time-reversal-breaking phase $\phi$ on a second bond family, whose sign depends on the orientation of the hop. A staggered onsite potential $\pm M$ distinguishes the two sublattices:

$$
H = -t_1 \sum_{\langle i,j \rangle} c_i^\dagger c_j
    -t_2 \sum_{\langle\langle i,j \rangle\rangle} e^{i \nu_{ij} \phi}\, c_i^\dagger c_j
    + M \sum_i \xi_i\, c_i^\dagger c_i,
$$

where $\xi_i = +1$ on sublattice $A$ and $-1$ on $B$, and $\nu_{ij} = \pm 1$ encodes the chirality of the hop. The competition between the staggered mass $M$ and the time-reversal-breaking flux drives a topological phase transition between a trivial insulator and quantum-Hall phases with Chern number $\nu = \pm 1$. Because it is non-interacting, the Haldane model is efficiently solvable by exact diagonalization; due to that and its interesting physical properties and phase diagrams, we use it as a canonical reference for our library.

## Lattices

| Registry name | Lattice | Geometry |
| --- | --- | --- |
| `haldane-honeycomb` | honeycomb | $\phi$ rides the next-nearest-neighbor hop $t_2 e^{i\nu_{ij}\phi}$ within each triangular sublattice; three nearest neighbors, six next-nearest. |
| `haldane-square`    | square (checkerboard) | $\phi$ rides the nearest-neighbor hop $t_1 e^{\pm i\phi}$, staggered so the two triangles of each crossed square enclose opposite flux; $t_2$ is a real next-nearest-neighbor hop, along $x$ on $A$ and along $y$ on $B$. |

Both are two-sublattice lattices with a staggered mass, and both realize $\nu = \pm 1$ Chern bands, but the flux has to live on a bond family whose loops are *odd*. On the honeycomb lattice those are the next-nearest-neighbor triangles, which is Haldane's original construction. On the square lattice the next-nearest-neighbor loops are four-sided and always enclose zero net flux, so the square variant is built on the checkerboard geometry instead: the crossed squares make triangles out of two nearest-neighbor bonds and one next-nearest-neighbor bond, and the flux is carried by the nearest-neighbor bonds. This is the standard checkerboard-lattice Chern insulator.

That difference shows up in the phase boundary:

| Lattice | Topological region | Bulk gap |
| --- | --- | --- |
| honeycomb | $\lvert M\rvert < 3\sqrt{3}\,t_2\,\lvert\sin\phi\rvert$ | $2\,\lvert\, 3\sqrt{3}\,t_2 \sin\phi - \lvert M\rvert\, \rvert$, widest at $\phi = \pm\pi/2$ |
| square    | $\lvert M\rvert < 2\,t_2$, for $\phi \notin \{0, \pi/2, \pi\}$ | $2\,(2t_2 - \lvert M\rvert)$, flat in $\phi$ away from the degenerate fluxes |

On the square lattice the Dirac points sit at $(\pi, 0)$ and $(0, \pi)$ and the mass that gaps them is $M \mp 2t_2$, so the boundary is a $\phi$-independent pair of lines $\lvert M\rvert = 2t_2$; the Chern number flips sign with $\phi$ across $\pi/2$. The degenerate fluxes $\phi = 0, \pi/2, \pi$ leave the model gapless for $\lvert M\rvert < 2t_2$, since one of the two off-diagonal components of the Bloch Hamiltonian vanishes identically there. Because the Dirac points are at the Brillouin-zone edge, a finite-lattice run only resolves the gap closing when `Lx` and `Ly` are even.

## Parameters

| Parameter | Symbol | Meaning |
| --- | --- | --- |
| `t1`  | $t_1$  | Nearest-neighbor hopping ($A \leftrightarrow B$); carries $e^{\pm i\phi}$ on the square lattice |
| `t2`  | $t_2$  | Next-nearest-neighbor hopping magnitude ($A \to A$, $B \to B$); carries $e^{i\nu_{ij}\phi}$ on the honeycomb lattice |
| `phi` | $\phi$ | Haldane flux phase |
| `M`   | $M$    | Staggered sublattice mass ($+M$ on $A$, $-M$ on $B$) |

## Typical Sweep

The canonical phase diagram plots $M$ against $\phi$ at a fixed $t_2$. On the honeycomb lattice the zero-field quantum-Hall phases occupy the region $|M/t_2| < 3\sqrt{3}\,\left|\sin\phi\right|$, so the phase boundary appears as two sinusoidal lobes symmetric about $\phi = 0$, with peaks at $\pm 3\sqrt{3}\,t_2$. On the square lattice the same sweep gives a rectangular topological region $|M| < 2t_2$, pinched shut on the vertical lines $\phi = 0, \pi/2, \pi$. Reading the spectral gap instead of the ground-state energy renders either boundary sharply as a zero-gap curve.

## Example

The following sweeps $\phi$ against $M$ at $t_2 = 0.1$ on a $3\times 3$ honeycomb lattice and reads off the spectral gap. The underlying specification lives in `qbp/models/haldane-honeycomb.yaml`.

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
    model="haldane-honeycomb",
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

The same sweep on the square lattice needs only the model name changed (`qbp/models/haldane-square.yaml`). Here the topological region is the rectangle $|M| < 2t_2 = 1$, so an even lattice is used to put the Dirac points $(\pi, 0)$ and $(0, \pi)$ on the momentum grid:

```{jupyter-execute}
result = qbp.run(
    model="haldane-square",
    method=[Method.ANALYTIC],
    lattice=(4, 4),
    x_param="phi",
    x_range=(0.0, math.pi, math.pi / 50),
    y_param="M",
    y_range=(-2.0, 2.0, 0.05),
    model_params={"t1": 1.0, "t2": 0.5},
    observable="gap",
    heatmap=True,
)
```
