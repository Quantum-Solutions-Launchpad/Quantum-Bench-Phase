# End-to-End Workflow

## QBP Pipeline

A typical QBP run follows the same pipeline regardless of the model or algorithm:

<div class="qbp-pipeline">
  <div class="step" style="grid-area: 1 / 1;">Model</div>
  <div class="arrow arrow-right" style="grid-area: 1 / 2;"><svg viewBox="0 0 100 60" preserveAspectRatio="none" aria-hidden="true"><polygon points="0,18 60,18 60,0 100,30 60,60 60,42 0,42"/></svg></div>
  <div class="step" style="grid-area: 1 / 3;">Mapper</div>
  <div class="arrow arrow-down" style="grid-area: 2 / 3;"><svg viewBox="0 0 60 100" preserveAspectRatio="none" aria-hidden="true"><polygon points="18,0 42,0 42,60 60,60 30,100 0,60 18,60"/></svg></div>
  <div class="inject" style="grid-area: 3 / 1;">Qubit-Mapped<br>Hamiltonians</div>
  <div class="arrow arrow-right" style="grid-area: 3 / 2;"><svg viewBox="0 0 100 60" preserveAspectRatio="none" aria-hidden="true"><polygon points="0,18 60,18 60,0 100,30 60,60 60,42 0,42"/></svg></div>
  <div class="step" style="grid-area: 3 / 3;">Simulation</div>
  <div class="arrow arrow-down" style="grid-area: 4 / 3;"><svg viewBox="0 0 60 100" preserveAspectRatio="none" aria-hidden="true"><polygon points="18,0 42,0 42,60 60,60 30,100 0,60 18,60"/></svg></div>
  <div class="step" style="grid-area: 5 / 1;">Plot</div>
  <div class="arrow arrow-left" style="grid-area: 5 / 2;"><svg viewBox="0 0 100 60" preserveAspectRatio="none" aria-hidden="true"><polygon points="100,18 40,18 40,0 0,30 40,60 40,42 100,42"/></svg></div>
  <div class="step" style="grid-area: 5 / 3;">Result</div>
</div>

- **Model.** A tight-binding Hamiltonian on a chosen lattice with chosen parameters.
- **Mapper.** A fermion-to-qubit mapping (e.g. Jordan-Wigner) that turns the Hamiltonian into Pauli operators.
- **Simulation.** A quantum algorithm — VQE (a parameterized ansatz optimized by a classical optimizer) or IQPE (a Trotterized time-evolution circuit) — or a classical tensor-network benchmark (DMRG), each estimating ground-state energies or other observables across the sweep.
- **Result.** Ground-state energies, gaps, or other observables across a parameter sweep.
- **Plot.** A line plot, 3D plot, or heatmap of the observable.
- **Qubit-Mapped Hamiltonians.** Instead of starting from a built-in model and mapper, you can inject a pre-mapped qubit Hamiltonian — for example a [HamLib](https://quantum-journal.org/papers/q-2024-12-11-1559/) problem Hamiltonian or your own Pauli operators — straight into the simulation stage.

## Case Study: Haldane Model

Suppose you want to explore the phase diagrams of the [Haldane model](https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.61.2015). It lives on a honeycomb lattice with two sublattices, $A$ and $B$. Electrons hop between nearest-neighbor sites with amplitude $t_1$ and between next-nearest-neighbor sites with a complex amplitude $t_2 e^{\pm i \phi}$, whose sign depends on the orientation of the hopping. A staggered onsite potential $\pm M$ distinguishes the two sublattices.

<figure class="qbp-figure qbp-figure--tall">
  <svg class="qbp-diagram" viewBox="180 34 260 288" role="img" aria-labelledby="haldane-fig-title haldane-fig-desc">
    <title id="haldane-fig-title">The Haldane model on the honeycomb lattice</title>
    <desc id="haldane-fig-desc">One hexagonal plaquette of a honeycomb lattice, with bonds continuing outward to the rest of the lattice. Its six sites alternate between the A and B sublattices. Solid bonds join nearest neighbors on opposite sublattices with amplitude t1. Two dashed triangles join next-nearest neighbors within each sublattice; arrows on them circulate counterclockwise and mark the direction in which a hop picks up phase plus phi. The A sublattice carries onsite energy plus M and the B sublattice minus M.</desc>
    <line class="dg-t1" x1="310" y1="90" x2="310" y2="48"/>
    <line class="dg-t1" x1="233.8" y1="134" x2="197.4" y2="113"/>
    <line class="dg-t1" x1="233.8" y1="222" x2="197.4" y2="243"/>
    <line class="dg-t1" x1="310" y1="266" x2="310" y2="308"/>
    <line class="dg-t1" x1="386.2" y1="222" x2="422.6" y2="243"/>
    <line class="dg-t1" x1="386.2" y1="134" x2="422.6" y2="113"/>
    <polygon class="dg-t1" points="310,90 233.8,134 233.8,222 310,266 386.2,222 386.2,134" stroke-linejoin="round"/>
    <polygon class="dg-nnn" points="310,90 233.8,222 386.2,222"/>
    <polygon class="dg-nnn" points="233.8,134 310,266 386.2,134"/>
    <path class="dg-arrow" d="M -7,-5.5 L 7,0 L -7,5.5 Z" transform="translate(271.9,156) rotate(120)"/>
    <path class="dg-arrow" d="M -7,-5.5 L 7,0 L -7,5.5 Z" transform="translate(310,222) rotate(0)"/>
    <path class="dg-arrow" d="M -7,-5.5 L 7,0 L -7,5.5 Z" transform="translate(348.1,156) rotate(-120)"/>
    <path class="dg-arrow" d="M -7,-5.5 L 7,0 L -7,5.5 Z" transform="translate(271.9,200) rotate(60)"/>
    <path class="dg-arrow" d="M -7,-5.5 L 7,0 L -7,5.5 Z" transform="translate(348.1,200) rotate(-60)"/>
    <path class="dg-arrow" d="M -7,-5.5 L 7,0 L -7,5.5 Z" transform="translate(310,134) rotate(180)"/>
    <circle class="dg-site-a" cx="310" cy="90" r="13"/>
    <circle class="dg-site-b" cx="233.8" cy="134" r="13"/>
    <circle class="dg-site-a" cx="233.8" cy="222" r="13"/>
    <circle class="dg-site-b" cx="310" cy="266" r="13"/>
    <circle class="dg-site-a" cx="386.2" cy="222" r="13"/>
    <circle class="dg-site-b" cx="386.2" cy="134" r="13"/>
    <text class="dg-in-a" x="310" y="90" dy="0.34em">A</text>
    <text class="dg-in-b" x="233.8" y="134" dy="0.34em">B</text>
    <text class="dg-in-a" x="233.8" y="222" dy="0.34em">A</text>
    <text class="dg-in-b" x="310" y="266" dy="0.34em">B</text>
    <text class="dg-in-a" x="386.2" y="222" dy="0.34em">A</text>
    <text class="dg-in-b" x="386.2" y="134" dy="0.34em">B</text>
    <text class="dg-label" x="338" y="76" text-anchor="start">+<tspan class="dg-var">M</tspan></text>
    <text class="dg-label" x="338" y="286" text-anchor="start">&#8722;<tspan class="dg-var">M</tspan></text>
  </svg>
  <ul class="qbp-legend">
    <li><span class="sw sw-t1"></span><span>nearest neighbor, <em>t</em><sub>1</sub></span></li>
    <li><span class="sw sw-t2"></span><span>next-nearest neighbor, <em>t</em><sub>2</sub>e<sup><em>i&phi;</em></sup></span></li>
  </ul>
  <figcaption>One plaquette of the honeycomb lattice, with bonds continuing out into the rest of the lattice. Solid bonds connect the two sublattices with amplitude <em>t</em><sub>1</sub>, and the filled <em>A</em> and open <em>B</em> sites carry the staggered onsite energies &plusmn;<em>M</em>. The two dashed triangles are the next-nearest-neighbor hoppings, which stay within a single sublattice: a hop taken <em>along</em> an arrow picks up the phase <em>e</em><sup>+<em>i&phi;</em></sup> (that is, <em>&nu;<sub>ij</sub></em> = +1), and a hop against one picks up <em>e</em><sup>&minus;<em>i&phi;</em></sup>.</figcaption>
</figure>

Its tight-binding Hamiltonian can be written as

$$
H = -t_1 \sum_{\langle i,j \rangle} c_i^\dagger c_j
    -t_2 \sum_{\langle\langle i,j \rangle\rangle} e^{i \nu_{ij} \phi}\, c_i^\dagger c_j
    + M \sum_i \xi_i\, c_i^\dagger c_i,
$$

where $\xi_i = +1$ on sublattice $A$ and $-1$ on sublattice $B$, and $\nu_{ij} = \pm 1$ encodes the chirality of the next-nearest-neighbor hopping. The competition between the staggered mass $M$ and the time-reversal-breaking hopping $t_2$ drives a topological phase transition.

### Analytic Computation

The canonical phase diagram of the Haldane model is a plot of $M/t_2$ against $\phi$. In this setup, the phase boundaries appear as sinusoidal waves with peaks at $\pm 3\sqrt{3}$. As an example, let's fix $t_2 = 0.1$ and plot $M$ against $\phi$ as a heatmap using QBP. We will begin by just performing an analytic computation.

```{jupyter-execute}
:hide-code:

import io
import os
import sys
from pathlib import Path
from loguru import logger

logger.remove()
sys.stdout = sys.stderr = io.StringIO()

import qbp
from qbp import Method

def _find_data_dir() -> Path:
    for base in (Path.cwd(), *Path.cwd().parents):
        candidate = base / "docs" / "_data"
        if candidate.is_dir():
            return candidate
        candidate = base / "_data"
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError("docs/_data not found relative to cwd")

_DATA_DIR = _find_data_dir()

_real_run = qbp.run

def _patched_run(*args, **kwargs):
    methods = kwargs.get("method") or []
    names = {getattr(m, "value", m) for m in methods}
    if names & {"vqe", "iqpe"}:
        fname = (
            "simulated-noisy-3d-n_occ-vs-t2.json"
            if kwargs.get("backend")
            else "simulated-ideal-3d-n_occ-vs-t2.json"
        )
        result = qbp.load_result(str(_DATA_DIR / fname))
        result.plot(hide_plot=kwargs.get("hide_plot", False))
        return result
    return _real_run(*args, **kwargs)

qbp.run = _patched_run
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
    heatmap=True,
)
```

As you can see, we can clearly see the phase boundary as lower-energy states tracing sinusoidal waves that peak at $\pm 3\sqrt{3} \cdot 0.1 \approx 0.5$. 

You can use the `observable` parameter to plot different observables. By default, ground-state energy ($E$) is used, but you can set a handful of different parameters. Let's try plotting the spectral gap ($\Delta_\text{gap}$) instead.

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

Here, the same phase boundary is far more visible as a zero-gap area. Depending on the properties of the model you're investigating, different observables may be better suited to see different dynamic effects.

### Ideal Simulation

The real power of QBP, though, comes from comparing these analytic diagonalization runs against observables computed by quantum algorithms. The same `qbp.run` entry point drives them: pass quantum methods such as `Method.VQE` and `Method.IQPE` in the `method` list, and add `Method.ANALYTIC` to overlay the exact diagonalization surface as a reference. Per-method settings go in `method_params`, keyed by the method. Leaving `backend` unset runs on a noise-free statevector simulator. The call returns a `RunResult` that already knows how to plot itself.

Let's stick with the Haldane model but switch axes: fix $\phi = \pi/4$ and $M = 0.1$, and sweep occupation number $N_\text{occ}$ against $t_2$. The code to compute the ground-state energies for this configuration with a noise-free simulator looks as follows:

```{jupyter-execute}
import math
import qbp
from qbp import Method

result = qbp.run(
    model="haldane",
    method=[Method.ANALYTIC, Method.VQE, Method.IQPE],
    lattice=(2, 2),
    x_param="n_occ",
    y_param="t2",
    y_range=(0.0, 1.0, 0.25),
    model_params={"t1": 1.0, "phi": math.pi / 4, "M": 0.1},
    method_params={
        Method.VQE: {"iters": 50, "layers": 1, "reps": 1},
        Method.IQPE: {"time": 0.1, "trot": 1, "iters": 1, "reps": 1},
    },
)
```

The analytic surface is the smooth diagonalization baseline. VQE and IQPE samples sit on top of it. If the simulation is accurate, they should hug the baseline closely, with residual gaps coming from finite ansatz depth (`layers`) or limited phase-estimation precision (`iters`). Cranking those knobs up tightens agreement at the cost of runtime.

### Noisy Simulation

Real hardware, especially today's quantum hardware, is far noisier than this, though. Passing a `backend` keeps the same sweep shape but routes the circuits through a noise model. Here we use a fake snapshot of IBM's Sherbrooke device, but you can pass any Qiskit backend through the `backend` keyword to use your own.

```{jupyter-execute}
import math
import qbp
from qbp import Method

result = qbp.run(
    model="haldane",
    method=[Method.ANALYTIC, Method.VQE, Method.IQPE],
    lattice=(2, 2),
    x_param="n_occ",
    y_param="t2",
    y_range=(0.0, 1.0, 0.25),
    model_params={"t1": 1.0, "phi": math.pi / 4, "M": 0.1},
    method_params={
        Method.VQE: {"iters": 50, "layers": 1, "reps": 1},
        Method.IQPE: {"time": 0.1, "trot": 1, "iters": 1, "reps": 1},
    },
    backend="FakeSherbrooke",
)
```

The analytic surface is unchanged, but the VQE and IQPE markers drift away from it—IQPE tends to scatter more aggressively because its single-shot phase readout amplifies gate errors, while VQE's variational averaging hides some of the noise but biases upward.