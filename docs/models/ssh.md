# SSH

The **Su–Schrieffer–Heeger (SSH) model** is a one-dimensional bipartite chain with two sites, $A$ and $B$, per unit cell and staggered nearest-neighbor hopping. Electrons hop within a cell with amplitude $t_1$ and between adjacent cells with amplitude $t_2$:

$$
H = -t_1 \sum_{n} |n, B\rangle\langle n, A|
    -t_2 \sum_{n} |n+1, A\rangle\langle n, B|
    + \text{h.c.}
$$

The ratio of the two hoppings drives a topological phase transition: the chain is trivial when $t_1 > t_2$ and topological when $t_2 > t_1$, with the bulk gap closing exactly on the line $|t_2/t_1| = 1$. It is the simplest model in the catalog and a natural first stop for learning the [`qbp.run`](../api/runners.md) workflow.

## Parameters

| Parameter | Symbol | Meaning |
| --- | --- | --- |
| `t1` | $t_1$ | Intra-cell hopping ($A \leftrightarrow B$ within a cell) |
| `t2` | $t_2$ | Inter-cell hopping ($B \to A$ in the next cell) |

## Typical Sweep

The canonical phase diagram is the spectral gap over the $(t_1, t_2)$ plane. The gap collapses to zero along the diagonal $t_1 = t_2$, tracing the boundary between the trivial ($t_1 > t_2$) and topological ($t_2 > t_1$) phases as a dark valley. Sweeping the filling `n_occ` against $t_2$ instead exposes how the same energy landscape deforms as the chain crosses that transition.

## Example

The following reproduces the SSH phase diagram as a spectral-gap heatmap on an eight-cell chain. The underlying specification lives in `qbp/models/ssh.yaml`.

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
    model="ssh",
    method=[Method.ANALYTIC],
    lattice=(8,),
    x_param="t1",
    x_range=(0.0, 2.0, 0.02),
    y_param="t2",
    y_range=(0.0, 2.0, 0.02),
    observable="gap",
    heatmap=True,
)
```

The dark diagonal is the gap-closing line $t_1 = t_2$ that separates the two phases.
