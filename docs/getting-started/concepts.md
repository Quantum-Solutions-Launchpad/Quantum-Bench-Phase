# Concepts

This page introduces the physics and algorithms that QuaPh is designed around. We use the [Haldane model](https://en.wikipedia.org/wiki/Haldane_model) throughout as a running example, since it is the case study used in the accompanying paper and is a built-in model of the library.

## The Haldane Model

The **Haldane model** is a tight-binding model on a honeycomb lattice with two sublattices, $A$ and $B$. Electrons hop between nearest-neighbor sites with amplitude $t_1$ and between next-nearest-neighbor sites with amplitude $t_2 e^{\pm i\phi}$, where the sign depends on the orientation of the hop. A staggered onsite potential $\pm M$ distinguishes the two sublattices.

Its tight-binding Hamiltonian is

$$
H = -t_1 \sum_{\langle i,j \rangle} c_i^\dagger c_j
  - t_2 \sum_{\langle\langle i,j \rangle\rangle} e^{i \nu_{ij} \phi} c_i^\dagger c_j
  + M \sum_i \xi_i c_i^\dagger c_i,
$$

where $\xi_i = \pm 1$ labels the sublattice, $\nu_{ij} = \pm 1$ encodes the chirality of the next-nearest-neighbor hop, and $c_i^\dagger, c_j$ are fermionic creation and annihilation operators.

QuaPh treats the Haldane model as an $m \times n$ honeycomb lattice with **periodic boundary conditions (PBCs)**, where $m$ and $n$ count unit cells along each direction. A parallelogram slice of the lattice is used, with edges wrapping around to the opposite side. Open boundary conditions are also supported.

## Glossary of Relevant Terms

**Tight-Binding Hamiltonian**
: A quadratic Hamiltonian written in terms of fermionic operators $c_i^\dagger c_j$ describing electrons hopping between lattice sites. Because it is quadratic, it can be diagonalized in the single-particle basis, giving $2N$ eigenmodes with energies $\varepsilon_1 \leq \varepsilon_2 \leq \cdots \leq \varepsilon_{2N}$ on a lattice with $N = mn$ unit cells.

**Occupation Number (`n_occ`)**
: The number of single-particle modes that are filled. Since $H$ commutes with the total particle number, the many-body Hilbert space splits into sectors labeled by $N_\text{occ} \in \{0, 1, \dots, 2N\}$. Within each sector the ground state fills the $N_\text{occ}$ lowest modes, giving energy $E(N_\text{occ}) = \sum_{k=1}^{N_\text{occ}} \varepsilon_k$. The canonical Haldane phase diagram corresponds to half-filling, $N_\text{occ} = N$; other fillings probe qualitatively different ground states of the same Hamiltonian.

**Phase-Diagram Sweep (`x_param` / `y_param`)**
: A two-dimensional scan over a chosen pair of parameters, with all others held fixed. For the Haldane model, the canonical sweep is $M/t_2$ against $\phi$ at half-filling, which exhibits zero-field quantum Hall phases inside the region $|M/t_2| < 3\sqrt{3} \left|\sin\phi\right|$. Other sweeps, such as $N_\text{occ}$ against $t_2$, expose filling-dependent structure that is invisible in the canonical diagram.

**Bloch vs. Real-Space Mode**
: Bloch mode exploits translation invariance under PBCs to diagonalize the Hamiltonian one momentum block at a time, which is fast but only available with PBCs. Real-space mode works directly with site operators, supports open boundaries, and is required when translation symmetry is broken.

**Fermion-to-Qubit Mapping**
: A protocol that rewrites fermionic operators as Pauli operators so the Hamiltonian can run on a quantum computer. QuaPh uses mappings such as [Jordan-Wigner](https://en.wikipedia.org/wiki/Jordan%E2%80%93Wigner_transformation) that send one spin-orbital to exactly one qubit, so the qubit count equals the number of sites.

## Quantum Simulation Techniques

QuaPh supports two quantum algorithms for estimating ground-state energies.

**Variational Quantum Eigensolver (VQE)**
: A hybrid quantum-classical algorithm that exploits the variational principle $E_0 \leq \langle\Psi|\hat{H}|\Psi\rangle / \langle\Psi|\Psi\rangle$. The trial state $|\Psi\rangle = U(\boldsymbol{\theta})|\boldsymbol{0}\rangle$ is prepared by a parameterized circuit (the **ansatz**), and a classical optimizer iteratively updates $\boldsymbol{\theta}$ to minimize the measured energy. Each iteration uses a shallow circuit, which makes VQE well suited to NISQ-era hardware. The user chooses the qubit mapping, the ansatz (e.g. an excitation-preserving ansatz of two-qubit gates), and the classical optimizer (e.g. Adam or SPSA).

**Iterative Quantum Phase Estimation (IQPE)**
: A variant of [quantum phase estimation](https://en.wikipedia.org/wiki/Quantum_phase_estimation_algorithm) that trades qubit count for iteration count. Phase estimation extracts $\theta$ from $U|\psi\rangle = e^{2\pi i\theta}|\psi\rangle$; precision in IQPE is controlled by the number of iterations $m$, which determines the bits $\phi = 0.x_1 x_2 \cdots x_m$ of the phase. Because $\hat{H}$ is not unitary, IQPE simulates the time-evolution operator $U(t) = e^{-i\hat{H}t}$, decomposed into gates via **Trotterization**. The user chooses the evolution time $t$ (typically so that $E_\text{max} t < 2\pi$), the number of Trotter steps $N_\text{trot}$, and the number of iterations $m$. IQPE circuits are deeper than VQE circuits, but when they are viable they are typically more accurate.

**Analytic, simulated, and noisy runs**
: Every QuaPh workflow can be executed in three modes. *Analytic* runs diagonalize the Hamiltonian classically and return the exact answer; they are the fastest and serve as a ground-truth reference. *Simulated ideal* runs execute the quantum circuit on a noiseless statevector simulator, so they capture algorithmic error (e.g. Trotter or ansatz error) without hardware noise. *Simulated noisy* runs use a noise model (or a real backend via Qiskit IBM Runtime) and additionally capture device noise.

## Pipeline Overview

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