# Kane–Mele

The **Kane–Mele model** is the time-reversal-symmetric, spinful cousin of the Haldane model on the honeycomb lattice. It keeps the two sublattices $A$ and $B$ and the staggered mass $M$, but replaces the Haldane flux with a spin-dependent next-nearest-neighbor spin–orbit coupling $i\lambda_{SO}\,\nu_{ij}\,\sigma_z$:

$$
H = -t \sum_{\langle i,j\rangle,\sigma} c_{i,\sigma}^\dagger c_{j,\sigma}
    + i\lambda_{SO} \sum_{\langle\langle i,j\rangle\rangle,\sigma\sigma'} \nu_{ij}\, (\sigma_z)_{\sigma\sigma'}\, c_{i,\sigma}^\dagger c_{j,\sigma'}
    + M \sum_i \xi_i\, c_i^\dagger c_i.
$$

Spin-up and spin-down electrons see opposite effective Haldane fluxes, so the model preserves time-reversal symmetry and realizes a $\mathbb{Z}_2$ topological insulator rather than a quantum-Hall phase. The spin–orbit term competes with the staggered mass, driving a transition between the $\mathbb{Z}_2$ topological insulator and the trivial insulator at $M = 3\sqrt{3}\,\lambda_{SO}$.

## Parameters

| Parameter   | Symbol           | Meaning |
| --- | --- | --- |
| `t`         | $t$              | Nearest-neighbor hopping ($A \leftrightarrow B$) |
| `lambda_SO` | $\lambda_{SO}$   | Next-nearest-neighbor spin–orbit coupling |
| `M`         | $M$              | Staggered sublattice mass ($+M$ on $A$, $-M$ on $B$) |

## Typical Sweep

The canonical phase diagram plots $\lambda_{SO}$ against $M$ at fixed $t$. The topological ($\mathbb{Z}_2$) and trivial insulators are separated by the line $M = 3\sqrt{3}\,\lambda_{SO}$, across which the bulk gap closes and reopens. Plotting the spectral gap makes that boundary appear as a sharp zero-gap ridge.

## Example

The following sweeps $\lambda_{SO}$ against $M$ on a $3\times 3$ honeycomb lattice at $t = 1$ and reads off the spectral gap. The underlying specification lives in `qbp/models/kane_mele.yaml`.

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
    model="kane-mele",
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
