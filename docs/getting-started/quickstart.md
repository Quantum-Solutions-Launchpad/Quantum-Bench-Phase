# Quickstart

In short, using QuaPh involves the following three steps:

1. Selecting the model you'd like to compute over.
2. Setting your model parameters, both parameters to fix and parameters to sweep over.
3. Selecting the technique you'd like to use: analytic computation or quantum simulation.

Consider the [Su–Schrieffer–Heeger model](https://en.wikipedia.org/wiki/Su%E2%80%93Schrieffer%E2%80%93Heeger_model) as an example. Each unit cell in the SSH model has two sites, $A$ and $B$, and each electron can hop to the opposite site in its unit cell or to the opposite site in its adjacent unit cell.

```{eval-rst}
.. todo:: Add SSH diagram here (made using TikZ)
```

Its tight-binding Hamiltonian can be written as

$$
H = t_1 \sum_{n=1}^N |n,B\rangle \langle n,A|
  + t_2 \sum_{n=1}^N |n+1,A\rangle \langle n,B|
  + \text{h.c.}
$$

where $\text{h.c.}$ denotes the [Hermitian conjugate](https://en.wikipedia.org/wiki/Hermitian_conjugate), $t_1$ denotes the intra-cell hopping, and $t_2$ denotes the inter-cell hopping. The relative size of $t_1$ and $t_2$ controls a topological phase transition: the chain is trivial when $t_1 > t_2$ and topological when $t_2 > t_1$.

Suppose you want to see render the phase diagram of the SSH model, physically observing the phase boundary $t_1=t_2$. A natural observable to plot is the *spectral gap*, the energy difference between the ground state and the first excited state. The gap vanishes exactly on the line $t_1 = t_2$ because the two SSH bands touch there, marking the bulk band closing that separates the trivial and topological phases.

Since the SSH model is built into QuaPh, you can execute it using one command after importing the library into Python. You simply need to decide the size of the lattice, which in this case is a one-dimensional chain, and the bounds and step size of the parameters to sweep over. 

The following code shows the generation of the phase diagram for the SSH model formatted as a heatmap with $t_1$ on the $x$-axis, $t_2$ on the $y$-axis, and the spectral gap as the observable to calculate in the form of a heatmap.

```{jupyter-execute}
:hide-code:

import io
import sys
from loguru import logger

logger.remove()
sys.stdout = sys.stderr = io.StringIO()
```

```{jupyter-execute}
import quaph

result = quaph.run_analytic(
    model="ssh",
    lattice=(8,),
    x_param="t1",
    x_range=(0.0, 2.0, 0.02),
    y_param="t2",
    y_range=(0.0, 2.0, 0.02),
    observable="gap",
    heatmap=True,
)
```

You can use QuaPh to run far more powerful workloads, particularly ones that involve quantum simulation. You can look at the [End-to-End Workflow](../user-guide/workflow.md) page to get a sense of a full example using a more complicated tight-binding model.