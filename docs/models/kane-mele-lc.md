# Kane–Mele (Liu–Chen)

The **Kane–Mele (Liu–Chen) model** generalizes the [Kane–Mele model](kane-mele.md) by attaching a tunable Haldane-like phase $\phi$ to the spin–orbit hopping. Spin-up electrons pick up $\lambda_{SO}\, e^{+i\phi\nu_{ij}}$ on the spin–orbit bond and spin-down electrons pick up the complex conjugate:

$$
H = -t \sum_{\langle i,j\rangle,\sigma} c_{i,\sigma}^\dagger c_{j,\sigma}
    + \lambda_{SO} \sum_{\langle\langle i,j\rangle\rangle,\sigma} e^{i\phi\,\nu_{ij}\, s_\sigma}\, c_{i,\sigma}^\dagger c_{j,\sigma}
    + M \sum_i \xi_i\, c_i^\dagger c_i,
$$

where $s_\sigma = +1$ for spin-up and $-1$ for spin-down. Setting $\phi = \pi/2$ recovers the standard imaginary spin–orbit coupling $i\lambda_{SO}\nu_{ij}\sigma_z$ of the [Kane–Mele model](kane-mele.md); away from $\phi = \pi/2$ the phase interpolates continuously, letting you tune the spin-dependent flux that protects the $\mathbb{Z}_2$ topological insulator.

## Lattices

| Registry name | Lattice | Spin–orbit bonds |
| --- | --- | --- |
| `kane-mele-lc-honeycomb` | honeycomb | $\lambda_{SO} e^{i\phi\nu_{ij}s_\sigma}$ on the next-nearest-neighbor hop |
| `kane-mele-lc-square`    | square (checkerboard) | $\lambda_{SO} e^{\pm i\phi}$ added to the nearest-neighbor hop, staggered by bond orientation; real next-nearest-neighbor $t_2$ along $x$ on $A$ and $y$ on $B$ |

Both variants reduce exactly to the corresponding [Kane–Mele](kane-mele.md) model at $\phi = \pi/2$, where the spin–orbit term becomes purely imaginary, and both become time-reversal-symmetric real hopping models at $\phi = 0$. The square variant sits on the checkerboard geometry for the same reason the [Haldane](haldane.md) one does—square next-nearest-neighbor loops enclose no net flux—and therefore carries the extra real hopping `t2`. There is no triangular variant: the staggered mass needs a two-sublattice cell.

## Parameters

| Parameter   | Symbol          | Meaning |
| --- | --- | --- |
| `t`         | $t$             | Nearest-neighbor hopping ($A \leftrightarrow B$) |
| `lambda_SO` | $\lambda_{SO}$  | Spin–orbit coupling magnitude (next-nearest-neighbor on honeycomb, nearest-neighbor on square) |
| `M`         | $M$             | Staggered sublattice mass ($+M$ on $A$, $-M$ on $B$) |
| `phi`       | $\phi$          | Generalized Haldane phase on the spin–orbit hop ($\pi/2$ recovers Kane–Mele) |
| `t2`        | $t_2$           | Real next-nearest-neighbor hopping (**square lattice only**) |

## Typical Sweep

On either lattice the canonical phase diagram plots $\phi$ against $M$ at fixed $t$ and $\lambda_{SO}$ (and fixed $t_2$ on the square lattice). The $\mathbb{Z}_2$ topological region is bounded by a sinusoidal curve in $\phi$, widest at $\phi = \pm\pi/2$ (where the model is fully Kane–Mele) and pinching shut at $\phi = 0, \pm\pi$ where the spin–orbit flux vanishes and the mass $M$ trivially gaps the spectrum. Plotting the spectral gap renders the boundary as a zero-gap curve. On the square lattice the $\mathbb{Z}_2$ region is instead the band $\lvert M\rvert < 2t_2$, independent of $\phi$, closing only where the nearest-neighbor amplitude $-t + \lambda_{SO}e^{i\phi}$ becomes real: at $\phi = 0, \pi$, and—when $\lambda_{SO} > t$—on the extra line $\cos\phi = t/\lambda_{SO}$.

## Example

The following sweeps $\phi$ against $M$ on a $3\times 3$ honeycomb lattice at $t = 1$ and $\lambda_{SO} = 0.1$, reading off the spectral gap. The underlying specification lives in `qbp/models/kane-mele-lc-honeycomb.yaml`; the square-lattice spec is `qbp/models/kane-mele-lc-square.yaml`, which takes the same sweep with an added `t2`.

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
    model="kane-mele-lc-honeycomb",
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
