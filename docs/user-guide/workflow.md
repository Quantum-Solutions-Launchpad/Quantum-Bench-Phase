# End-to-End Workflow

## QuaPh Pipeline

A typical QuaPh run follows the same pipeline regardless of the model or algorithm:

<div class="quaph-pipeline">
  <div class="pipeline-row">
    <div class="step">Model</div>
    <div class="arrow arrow-right"><svg viewBox="0 0 100 60" preserveAspectRatio="none" aria-hidden="true"><polygon points="0,18 60,18 60,0 100,30 60,60 60,42 0,42"/></svg></div>
    <div class="step">Mapper</div>
    <div class="arrow arrow-right"><svg viewBox="0 0 100 60" preserveAspectRatio="none" aria-hidden="true"><polygon points="0,18 60,18 60,0 100,30 60,60 60,42 0,42"/></svg></div>
    <div class="step">Ansatz /<br>Trotterization</div>
  </div>
  <div class="connector">
    <div class="arrow arrow-down"><svg viewBox="0 0 60 100" preserveAspectRatio="none" aria-hidden="true"><polygon points="18,0 42,0 42,60 60,60 30,100 0,60 18,60"/></svg></div>
  </div>
  <div class="pipeline-row">
    <div class="step">Plot</div>
    <div class="arrow arrow-left"><svg viewBox="0 0 100 60" preserveAspectRatio="none" aria-hidden="true"><polygon points="100,18 40,18 40,0 0,30 40,60 40,42 100,42"/></svg></div>
    <div class="step">Result</div>
    <div class="arrow arrow-left"><svg viewBox="0 0 100 60" preserveAspectRatio="none" aria-hidden="true"><polygon points="100,18 40,18 40,0 0,30 40,60 40,42 100,42"/></svg></div>
    <div class="step">Optimizer</div>
  </div>
</div>

- **Model.** A tight-binding Hamiltonian on a chosen lattice with chosen parameters.
- **Mapper.** A fermion-to-qubit mapping (e.g. Jordan-Wigner) that turns the Hamiltonian into Pauli operators.
- **Ansatz / Trotterization.** A parameterized circuit (VQE) or a Trotterized time-evolution circuit (IQPE).
- **Optimizer.** A classical optimizer that updates ansatz parameters (VQE) or iterative execution of a quantum circuit (IQPE).
- **Result.** Ground-state energies, gaps, or other observables across a parameter sweep.
- **Plot.** A line plot, 3D plot, or heatmap of the observable.

## Case Study: Haldane Model

Suppose you want to explore the phase diagrams of the [Haldane model](https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.61.2015). It lives on a honeycomb lattice with two sublattices, $A$ and $B$. Electrons hop between nearest-neighbor sites with amplitude $t_1$ and between next-nearest-neighbor sites with a complex amplitude $t_2 e^{\pm i \phi}$, whose sign depends on the orientation of the hopping. A staggered onsite potential $\pm M$ distinguishes the two sublattices.

```{eval-rst}
.. todo:: Add Haldane lattice diagram here (made using TikZ)
```

Its tight-binding Hamiltonian can be written as

$$
H = -t_1 \sum_{\langle i,j \rangle} c_i^\dagger c_j
    -t_2 \sum_{\langle\langle i,j \rangle\rangle} e^{i \nu_{ij} \phi}\, c_i^\dagger c_j
    + M \sum_i \xi_i\, c_i^\dagger c_i,
$$

where $\xi_i = +1$ on sublattice $A$ and $-1$ on sublattice $B$, and $\nu_{ij} = \pm 1$ encodes the chirality of the next-nearest-neighbor hopping. The competition between the staggered mass $M$ and the time-reversal-breaking hopping $t_2$ drives a topological phase transition.

### Analytic Computation

The canonical phase diagram of the Haldane model is a plot of $M/t_2$ against $\phi$. In this setup, the phase boundaries appear as sinusoidal waves with peaks at $\pm 3\sqrt{3}$. As an example, let's fix $t_2 = 0.1$ and plot $M$ against $\phi$ as a heatmap using QuaPh. We will begin by just performing an analytic computation.

```{jupyter-execute}
:hide-code:

import io
import os
import sys
from pathlib import Path
from loguru import logger

logger.remove()
sys.stdout = sys.stderr = io.StringIO()

import quaph

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

def _patched_simulated_ideal(*args, **kwargs):
    result = quaph.load_result(str(_DATA_DIR / "simulated-ideal-3d-n_occ-vs-t2.json"))
    result.plot(hide_plot=kwargs.get("hide_plot", False))
    return result

def _patched_simulated_noisy(*args, **kwargs):
    result = quaph.load_result(str(_DATA_DIR / "simulated-noisy-3d-n_occ-vs-t2.json"))
    result.plot(hide_plot=kwargs.get("hide_plot", False))
    return result

quaph.run_simulated_ideal = _patched_simulated_ideal
quaph.run_simulated_noisy = _patched_simulated_noisy
```

```{jupyter-execute}
import math
import quaph

result = quaph.run_analytic(
    model="haldane",
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
import quaph

result = quaph.run_analytic(
    model="haldane",
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

The real power of QuaPh, though, comes from comparing these analytic diagonalization runs against observables computed by quantum algorithms. Two entry points wrap this functionality end-to-end: `quaph.run_simulated_ideal` runs VQE and IQPE on a noise-free statevector simulator, and `quaph.run_simulated_noisy` runs them through a noise model. Both use Qiskit circuitry to accomplish the simulation, and both return a `SimulatedResult` that already knows how to plot itself.

Let's stick with the Haldane model but switch axes: fix $\phi = \pi/4$ and $M = 0.1$, and sweep occupation number $N_\text{occ}$ against $t_2$. The code to compute the ground-state energies for this configuration with a noise-free simulator looks as follows:

```{jupyter-execute}
import math
import quaph

result = quaph.run_simulated_ideal(
    model="haldane",
    lattice=(2, 2),
    x_param="n_occ",
    y_param="t2",
    y_range=(0.0, 1.0, 0.25),
    model_params={"t1": 1.0, "phi": math.pi / 4, "M": 0.1},
    vqe_iters=50,
    vqe_layers=1,
    vqe_reps=1,
    iqpe_time=0.1,
    iqpe_trot=1,
    iqpe_iters=1,
    iqpe_reps=1,
)
```

The analytic surface is the smooth diagonalization baseline. VQE and IQPE samples sit on top of it. If the simulation is accurate, they should hug the baseline closely, with residual gaps coming from finite ansatz depth (`vqe_layers`) or limited phase-estimation precision (`iqpe_iters`). Cranking those knobs up tightens agreement at the cost of runtime.

### Noisy Simulation

Real hardware, especially today's quantum hardware, is far noisier than this, though. Swapping in `run_simulated_noisy` keeps the same sweep shape but routes the circuits through a noise model. By default, QuaPh uses a fake snapshot of IBM's Sherbrooke device, but you can pass any Qiskit backend through the `backend` keyword to use your own.

```{jupyter-execute}
import math
import quaph

result = quaph.run_simulated_noisy(
    model="haldane",
    lattice=(2, 2),
    x_param="n_occ",
    y_param="t2",
    y_range=(0.0, 1.0, 0.25),
    model_params={"t1": 1.0, "phi": math.pi / 4, "M": 0.1},
    vqe_iters=50,
    vqe_layers=1,
    vqe_reps=1,
    iqpe_time=0.1,
    iqpe_trot=1,
    iqpe_iters=1,
    iqpe_reps=1,
)
```

The analytic surface is unchanged, but the VQE and IQPE markers drift away from it—IQPE tends to scatter more aggressively because its single-shot phase readout amplifies gate errors, while VQE's variational averaging hides some of the noise but biases upward.