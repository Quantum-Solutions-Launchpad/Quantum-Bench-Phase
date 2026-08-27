# Kane–Mele

The **Kane–Mele model** is the time-reversal-symmetric, spinful cousin of the [Haldane model](haldane.md). It keeps the two sublattices $A$ and $B$ and the staggered mass $M$, but replaces the Haldane flux with a spin-dependent spin–orbit coupling $i\lambda_{SO}\,\nu_{ij}\,\sigma_z$:

$$
H = -t \sum_{\langle i,j\rangle,\sigma} c_{i,\sigma}^\dagger c_{j,\sigma}
    + i\lambda_{SO} \sum_{\langle\langle i,j\rangle\rangle,\sigma\sigma'} \nu_{ij}\, (\sigma_z)_{\sigma\sigma'}\, c_{i,\sigma}^\dagger c_{j,\sigma'}
    + M \sum_i \xi_i\, c_i^\dagger c_i.
$$

Spin-up and spin-down electrons see opposite effective Haldane fluxes, so the model preserves time-reversal symmetry and realizes a $\mathbb{Z}_2$ topological insulator rather than a quantum-Hall phase. The spin–orbit term competes with the staggered mass, driving a transition between the $\mathbb{Z}_2$ topological insulator and the trivial insulator at $M = 3\sqrt{3}\,\lambda_{SO}$.

## Lattices

| Registry name | Lattice | Spin–orbit bonds | Transition |
| --- | --- | --- | --- |
| `kane-mele-honeycomb` | honeycomb | $i\lambda_{SO}\nu_{ij}\sigma_z$ on the next-nearest-neighbor hop | $\lvert M\rvert = 3\sqrt{3}\,\lambda_{SO}$ |
| `kane-mele-square`    | square (checkerboard) | $\mp i\lambda_{SO}$ added to the nearest-neighbor hop, staggered by bond orientation; real next-nearest-neighbor $t_2$ along $x$ on $A$ and $y$ on $B$ | $\lvert M\rvert = 2\,t_2$ |

Each spin sector of a Kane–Mele model is a Haldane model, and the flux that gives it a Chern band must thread an odd loop of bonds. On the honeycomb lattice those loops are the next-nearest-neighbor triangles, so the spin–orbit term lives there. On the square lattice the next-nearest-neighbor loops are four-sided and enclose no net flux, so the square variant uses the checkerboard geometry: the spin–orbit term is the imaginary part of the nearest-neighbor hop, staggered so that the two triangles of each crossed square see opposite flux, and a real next-nearest-neighbor hopping `t2` supplies the momentum dependence that turns the spin–orbit flux into a gap. Spin-up and spin-down remain exact complex conjugates on both lattices, so both are time-reversal symmetric $\mathbb{Z}_2$ insulators.

The square variant therefore carries one extra parameter, `t2`, and its transition is driven by `t2` rather than by `lambda_SO`: any $\lambda_{SO} \neq 0$ opens the $\mathbb{Z}_2$ gap, and the mass closes it at $\lvert M \rvert = 2t_2$. There is no triangular variant: the staggered mass needs a two-sublattice cell.

## Parameters

| Parameter   | Symbol           | Meaning |
| --- | --- | --- |
| `t`         | $t$              | Nearest-neighbor hopping ($A \leftrightarrow B$) |
| `lambda_SO` | $\lambda_{SO}$   | Spin–orbit coupling (next-nearest-neighbor on honeycomb, nearest-neighbor on square) |
| `M`         | $M$              | Staggered sublattice mass ($+M$ on $A$, $-M$ on $B$) |
| `t2`        | $t_2$            | Real next-nearest-neighbor hopping (**square lattice only**) |

## Typical Sweep

On the honeycomb lattice the canonical phase diagram plots $\lambda_{SO}$ against $M$ at fixed $t$. The topological ($\mathbb{Z}_2$) and trivial insulators are separated by the line $M = 3\sqrt{3}\,\lambda_{SO}$, across which the bulk gap closes and reopens. Plotting the spectral gap makes that boundary appear as a sharp zero-gap ridge. On the square lattice the same diagram is flat in $\lambda_{SO}$—the boundary is the pair of horizontal lines $\lvert M \rvert = 2t_2$—so the informative sweep there is $t_2$ against $M$ at fixed $\lambda_{SO}$.

## Example

The following sweeps $\lambda_{SO}$ against $M$ on a $3\times 3$ honeycomb lattice at $t = 1$ and reads off the spectral gap. The underlying specification lives in `qbp/models/kane-mele-honeycomb.yaml`.

```{jupyter-execute}
:hide-code:

import io
import sys
from loguru import logger

logger.remove()
sys.stdout = sys.stderr = io.StringIO()
```

```{jupyter-execute}
import qbp
from qbp import Method

result = qbp.run(
    model="kane-mele-honeycomb",
    method=[Method.ANALYTIC],
    lattice=(3, 3),
    x_param="lambda_SO",
    x_range=(0.0, 0.4, 0.02),
    y_param="M",
    y_range=(-1.0, 1.0, 0.05),
    model_params={"t": 1.0},
    observable="gap",
    heatmap=True,
)
```

The zero-gap wedge tracks the transition line $M = 3\sqrt{3}\,\lambda_{SO}$, separating the trivial insulator (large $|M|$) from the $\mathbb{Z}_2$ topological insulator (large $\lambda_{SO}$).

On the square lattice the roles of the two couplings change, so sweep the next-nearest-neighbor hopping against the mass instead (`qbp/models/kane-mele-square.yaml`). An even lattice puts the Dirac points $(\pi, 0)$ and $(0, \pi)$ on the momentum grid, and a $2\times2$ cell grid (8 sites) is already enough to resolve the boundary exactly:

```{jupyter-execute}
result = qbp.run(
    model="kane-mele-square",
    method=[Method.ANALYTIC],
    lattice=(2, 2),
    x_param="t2",
    x_range=(0.0, 1.0, 0.02),
    y_param="M",
    y_range=(-1.5, 1.5, 0.05),
    model_params={"t": 1.0, "lambda_SO": 0.5},
    observable="gap",
    heatmap=True,
)
```

The zero-gap wedge here tracks $\lvert M \rvert = 2t_2$, with the $\mathbb{Z}_2$ insulator inside it and the trivial insulator outside.
