# Kane–Mele (Liu–Chen)

The **Kane–Mele (Liu–Chen) model** generalizes the [Kane–Mele model](kane-mele.md) by attaching a tunable Haldane-like phase $\phi$ to the next-nearest-neighbor spin–orbit hopping. Spin-up electrons pick up $\lambda_{SO}\, e^{+i\phi\nu_{ij}}$ on a next-nearest-neighbor hop and spin-down electrons pick up the complex conjugate:

$$
H = -t \sum_{\langle i,j\rangle,\sigma} c_{i,\sigma}^\dagger c_{j,\sigma}
    + \lambda_{SO} \sum_{\langle\langle i,j\rangle\rangle,\sigma} e^{i\phi\,\nu_{ij}\, s_\sigma}\, c_{i,\sigma}^\dagger c_{j,\sigma}
    + M \sum_i \xi_i\, c_i^\dagger c_i,
$$

where $s_\sigma = +1$ for spin-up and $-1$ for spin-down. Setting $\phi = \pi/2$ recovers the standard imaginary spin–orbit coupling $i\lambda_{SO}\nu_{ij}\sigma_z$ of the [Kane–Mele model](kane-mele.md); away from $\phi = \pi/2$ the phase interpolates continuously, letting you tune the spin-dependent flux that protects the $\mathbb{Z}_2$ topological insulator.

## Parameters

| Parameter   | Symbol          | Meaning |
| --- | --- | --- |
| `t`         | $t$             | Nearest-neighbor hopping ($A \leftrightarrow B$) |
| `lambda_SO` | $\lambda_{SO}$  | Next-nearest-neighbor spin–orbit coupling magnitude |
| `M`         | $M$             | Staggered sublattice mass ($+M$ on $A$, $-M$ on $B$) |
| `phi`       | $\phi$          | Generalized Haldane phase on the spin–orbit hop ($\pi/2$ recovers Kane–Mele) |

## Typical Sweep

The canonical phase diagram plots $\phi$ against $M$ at fixed $t$ and $\lambda_{SO}$. The $\mathbb{Z}_2$ topological region is bounded by a sinusoidal curve in $\phi$, widest at $\phi = \pm\pi/2$ (where the model is fully Kane–Mele) and pinching shut at $\phi = 0, \pm\pi$ where the spin–orbit flux vanishes and the mass $M$ trivially gaps the spectrum. Plotting the spectral gap renders the boundary as a zero-gap curve.

## Example

The following sweeps $\phi$ against $M$ on a $3\times 3$ honeycomb lattice at $t = 1$ and $\lambda_{SO} = 0.1$, reading off the spectral gap. The underlying specification lives in `qbp/models/kane_mele_lc.yaml`.

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
    model="kane-mele-lc",
    method=[Method.ANALYTIC],
    lattice=(3, 3),
    x_param="phi",
    x_range=(-math.pi, math.pi, math.pi / 50),
    y_param="M",
    y_range=(-1.0, 1.0, 0.05),
    model_params={"t": 1.0, "lambda_SO": 0.1},
    observable="gap",
    heatmap=True,
)
```

The topological region is widest near $\phi = \pm\pi/2$ and closes off toward $\phi = 0, \pm\pi$, where the spin–orbit flux can no longer protect the $\mathbb{Z}_2$ phase.
