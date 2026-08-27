# QuantumBenchPhase (QBP)

QBP is a library for the generation and benchmarking of quantum phase diagrams of topological and correlated lattice models as well as [HamLib](https://quantum-journal.org/papers/q-2024-12-11-1559/) problem Hamiltonians.

QBP supports six built-in tight binding models—each available on several lattices (honeycomb, square, triangular), registered as `<model>-<lattice>`—and offers an easy API to define your own custom Hamiltonians. Once you choose a model, you can easily run analytic computations generate phase diagrams sweeping over different parameters for a host of different observables, as well as benchmark variational quantum simulation techniques. Beyond its built-in models, QBP can also ingest pre-mapped qubit Hamiltonians directly from the [HamLib](https://quantum-journal.org/papers/q-2024-12-11-1559/) dataset, injecting them into the same simulation pipeline so you can sweep and benchmark those problem Hamiltonians exactly as you would a built-in model.

The library is packaged as both a Python API and a command-line interface. Both support the same features, so use the one you're more comfortable with.

## Installation

In order to install QBP, you will need Python version 3.10 or above.

`pip` is a tool for installing and managing Python packages from the [Python Package Index](https://pypi.org/). You can install QBP using `pip` via:

```bash
pip install qbp
```

### Dependencies

QBP relies on the following packages as dependencies, all of which can be installed via `pip`: [Qiskit](https://pypi.org/project/qiskit/), [Qiskit Nature](https://pypi.org/project/qiskit-nature/), [Qiskit Algorithms](https://pypi.org/project/qiskit-algorithms/), [NumPy](https://pypi.org/project/numpy/), [Matplotlib](https://pypi.org/project/matplotlib/), and [Loguru](https://pypi.org/project/loguru/).

## Quickstart

In short, using QBP involves the following three steps:

1. Selecting the model you'd like to compute over.
2. Setting your model parameters, both parameters to fix and parameters to sweep over.
3. Selecting the technique you'd like to use: analytic computation or quantum simulation.

Consider the [Su–Schrieffer–Heeger model](https://en.wikipedia.org/wiki/Su%E2%80%93Schrieffer%E2%80%93Heeger_model) as an example. Each unit cell in the SSH model has two sites, $A$ and $B$, and each electron can hop to the opposite site in its unit cell or to the opposite site in its adjacent unit cell.

Its tight-binding Hamiltonian can be written as

$$H = t_1 \sum_{n=1}^N \lvert n,B \rangle \langle n,A \rvert + t_2 \sum_{n=1}^N \lvert n+1,A \rangle \langle n,B \rvert + \text{h.c.}$$

where $\text{h.c.}$ denotes the [Hermitian conjugate](https://en.wikipedia.org/wiki/Hermitian_conjugate), $t_1$ denotes the intra-cell hopping, and $t_2$ denotes the inter-cell hopping. The relative size of $t_1$ and $t_2$ controls a topological phase transition: the chain is trivial when $t_1 > t_2$ and topological when $t_2 > t_1$.

Suppose you want to see render the phase diagram of the SSH model, physically observing the phase boundary $t_1=t_2$. A natural observable to plot is the *spectral gap*, the energy difference between the ground state and the first excited state. The gap vanishes exactly on the line $t_1 = t_2$ because the two SSH bands touch there, marking the bulk band closing that separates the trivial and topological phases.

Since the SSH model is built into QBP, you can execute it using one command after importing the library into Python. You simply need to decide the size of the lattice, which in this case is a one-dimensional chain, and the bounds and step size of the parameters to sweep over.

The following code shows the generation of the phase diagram for the SSH model formatted as a heatmap with $t_1$ on the $x$-axis, $t_2$ on the $y$-axis, and the spectral gap as the observable to calculate in the form of a heatmap.

```python
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

You can use QBP to run far more powerful workloads, particularly ones that involve quantum simulation. You can look at the documentation to get a sense of full examples using more complicated tight-binding models.

## Documentation

The full documentation is available at **[documentation URL]**.

It walks through the physics QBP is built around — the Haldane and Hubbard models, occupation number and phase-diagram sweeps, Bloch versus real-space modes, fermion-to-qubit mappings — and the algorithms behind each run: VQE, IQPE, and a DMRG tensor-network benchmark, each executable analytically, on an ideal statevector simulator, or under noise. From there it covers the full workflow: choosing an observable and sweeping it in one or two dimensions, tuning per-method parameters, running against a local noise model or real IBM and IQM hardware, inspecting and plotting a `RunResult`, and writing runs to disk to reload instead of recomputing.

It also documents the six built-in tight-binding models (SSH, Haldane, Kane–Mele, Kane–Mele-LC, Hubbard, and Haldane–Hubbard) with each one's Hamiltonian, lattices, parameters, and canonical sweep. Every 2D model ships on more than one lattice—`haldane-honeycomb` and `haldane-square`, `hubbard-honeycomb`, `hubbard-square` and `hubbard-triangular`, and so on—so a lattice change is a one-word change to the model name. The same pages cover three ways to define your own: a YAML spec, the tight-binding builder, or the full Python API. The remaining pages describe the components you configure a run with — observables, ansätze and initial states, qubit mappers, and classical optimizers — plus the CLI and interactive console, the complete API reference, and how to contribute.
