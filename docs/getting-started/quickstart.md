# Quickstart

In short, using QBP involves the following three steps:

1. Selecting the model you'd like to compute over.
2. Setting your model parameters, both parameters to fix and parameters to sweep over.
3. Selecting the technique you'd like to use: analytic computation or quantum simulation.

Consider the [Su–Schrieffer–Heeger model](https://en.wikipedia.org/wiki/Su%E2%80%93Schrieffer%E2%80%93Heeger_model) as an example. Each unit cell in the SSH model has two sites, $A$ and $B$, and each electron can hop to the opposite site in its unit cell or to the opposite site in its adjacent unit cell.

<figure class="qbp-figure">
  <svg class="qbp-diagram" viewBox="0 0 620 168" role="img" aria-labelledby="ssh-fig-title ssh-fig-desc">
    <title id="ssh-fig-title">The Su-Schrieffer-Heeger chain</title>
    <desc id="ssh-fig-desc">A one-dimensional chain of alternating A and B sites. Short bonds within each unit cell carry the intra-cell hopping amplitude t1; long bonds between neighboring unit cells carry the inter-cell hopping amplitude t2. A dashed box marks one unit cell, which contains one A site and one B site.</desc>
    <line class="dg-t2" x1="48" y1="90" x2="95" y2="90"/>
    <line class="dg-t2" x1="157" y1="90" x2="275" y2="90"/>
    <line class="dg-t2" x1="337" y1="90" x2="455" y2="90"/>
    <line class="dg-t2" x1="517" y1="90" x2="566" y2="90"/>
    <line class="dg-t1" x1="95" y1="90" x2="157" y2="90"/>
    <line class="dg-t1" x1="275" y1="90" x2="337" y2="90"/>
    <line class="dg-t1" x1="455" y1="90" x2="517" y2="90"/>
    <rect class="dg-cell" x="239" y="66" width="134" height="48" rx="10"/>
    <circle class="dg-dot" cx="18" cy="90" r="2.5"/>
    <circle class="dg-dot" cx="28" cy="90" r="2.5"/>
    <circle class="dg-dot" cx="38" cy="90" r="2.5"/>
    <circle class="dg-dot" cx="576" cy="90" r="2.5"/>
    <circle class="dg-dot" cx="586" cy="90" r="2.5"/>
    <circle class="dg-dot" cx="596" cy="90" r="2.5"/>
    <circle class="dg-site-a" cx="95" cy="90" r="14"/>
    <circle class="dg-site-b" cx="157" cy="90" r="14"/>
    <circle class="dg-site-a" cx="275" cy="90" r="14"/>
    <circle class="dg-site-b" cx="337" cy="90" r="14"/>
    <circle class="dg-site-a" cx="455" cy="90" r="14"/>
    <circle class="dg-site-b" cx="517" cy="90" r="14"/>
    <text class="dg-in-a" x="95" y="90" dy="0.34em">A</text>
    <text class="dg-in-b" x="157" y="90" dy="0.34em">B</text>
    <text class="dg-in-a" x="275" y="90" dy="0.34em">A</text>
    <text class="dg-in-b" x="337" y="90" dy="0.34em">B</text>
    <text class="dg-in-a" x="455" y="90" dy="0.34em">A</text>
    <text class="dg-in-b" x="517" y="90" dy="0.34em">B</text>
    <text class="dg-label" x="126" y="56"><tspan class="dg-var">t</tspan><tspan class="dg-sub" dy="5">1</tspan></text>
    <text class="dg-label" x="306" y="56"><tspan class="dg-var">t</tspan><tspan class="dg-sub" dy="5">1</tspan></text>
    <text class="dg-label" x="486" y="56"><tspan class="dg-var">t</tspan><tspan class="dg-sub" dy="5">1</tspan></text>
    <text class="dg-label" x="216" y="56"><tspan class="dg-var">t</tspan><tspan class="dg-sub" dy="5">2</tspan></text>
    <text class="dg-label" x="396" y="56"><tspan class="dg-var">t</tspan><tspan class="dg-sub" dy="5">2</tspan></text>
    <text class="dg-cell-label" x="306" y="138">unit cell <tspan class="dg-var">n</tspan></text>
  </svg>
  <figcaption>The SSH chain. Every unit cell holds one <em>A</em> site and one <em>B</em> site; the short intra-cell bond carries amplitude <em>t</em><sub>1</sub> and the long inter-cell bond carries <em>t</em><sub>2</sub>. Dimerizing the chain so that <em>t</em><sub>2</sub> &gt; <em>t</em><sub>1</sub> leaves an unpaired site at each end &mdash; the topological phase.</figcaption>
</figure>

Its tight-binding Hamiltonian can be written as

$$
H = t_1 \sum_{n=1}^N |n,B\rangle \langle n,A|
  + t_2 \sum_{n=1}^N |n+1,A\rangle \langle n,B|
  + \text{h.c.}
$$

where $\text{h.c.}$ denotes the [Hermitian conjugate](https://en.wikipedia.org/wiki/Hermitian_conjugate), $t_1$ denotes the intra-cell hopping, and $t_2$ denotes the inter-cell hopping. The relative size of $t_1$ and $t_2$ controls a topological phase transition: the chain is trivial when $t_1 > t_2$ and topological when $t_2 > t_1$.

Suppose you want to see render the phase diagram of the SSH model, physically observing the phase boundary $t_1=t_2$. A natural observable to plot is the *spectral gap*, the energy difference between the ground state and the first excited state. The gap vanishes exactly on the line $t_1 = t_2$ because the two SSH bands touch there, marking the bulk band closing that separates the trivial and topological phases.

Since the SSH model is built into QBP, you can execute it using one command after importing the library into Python. You simply need to decide the size of the lattice, which in this case is a one-dimensional chain, and the bounds and step size of the parameters to sweep over. 

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

You can use QBP to run far more powerful workloads, particularly ones that involve quantum simulation. You can look at the [End-to-End Workflow](../user-guide/workflow.md) page to get a sense of a full example using a more complicated tight-binding model.