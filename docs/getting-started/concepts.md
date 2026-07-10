# Concepts

This page introduces the physics and algorithms that QBP is designed around. The accompanying paper uses two built-in models as its case studies: the non-interacting [Haldane model](https://en.wikipedia.org/wiki/Haldane_model), which is efficiently solvable classically and serves as an exactly verifiable running example, and the interacting [Hubbard model](https://en.wikipedia.org/wiki/Hubbard_model), which is hard to solve classically and motivates quantum simulation.

## The Haldane Model

The **Haldane model** is a tight-binding model on a honeycomb lattice with two sublattices, $A$ and $B$. Electrons hop between nearest-neighbor sites with amplitude $t_1$ and between next-nearest-neighbor sites with amplitude $t_2 e^{\pm i\phi}$, where the sign depends on the orientation of the hop. A staggered onsite potential $\pm M$ distinguishes the two sublattices.

Its tight-binding Hamiltonian is

$$
H = -t_1 \sum_{\langle i,j \rangle} c_i^\dagger c_j
  - t_2 \sum_{\langle\langle i,j \rangle\rangle} e^{i \nu_{ij} \phi} c_i^\dagger c_j
  + M \sum_i \xi_i c_i^\dagger c_i,
$$

where $\xi_i = \pm 1$ labels the sublattice, $\nu_{ij} = \pm 1$ encodes the chirality of the next-nearest-neighbor hop, and $c_i^\dagger, c_j$ are fermionic creation and annihilation operators.

QBP treats the Haldane model as an $m \times n$ honeycomb lattice with **periodic boundary conditions (PBCs)**, where $m$ and $n$ count unit cells along each direction. A parallelogram slice of the lattice is used, with edges wrapping around to the opposite side. The canonical phase diagram plots $M/t_2$ against $\phi$: zero-field quantum Hall phases occur when $|M/t_2| < 3\sqrt{3}\left|\sin\phi\right|$, so the phase boundary appears as two sinusoidal lobes (Chern numbers $\nu = \pm 1$) symmetric about the $x$-axis, separated from the trivial insulator ($\nu = 0$).

Because the Haldane model is non-interacting, its ground-state energies can be found exactly by diagonalizing the single-particle Hamiltonian, so it is efficiently solvable classically. This makes it an ideal ground-truth reference and a natural segue to harder problems — its interacting cousin, the Hubbard model below, and its finite-geometry variants under open boundaries.

### Open Boundary Conditions and Quantum-Dot Geometries

Beyond the PBC phase diagrams, QBP also supports finite quantum-dot geometries for all models with **open boundary conditions (OBCs)**. Open boundaries break translational symmetry, so crystal momentum is no longer a good quantum number and the Bloch decomposition is unavailable; instead QBP builds and diagonalizes the full $N_s \times N_s$ single-particle Hamiltonian in the real-space site basis for a flake of $N_s$ retained sites. Two configurations are used in the main text:

- **Hard-wall dot.** An approximately circular flake obtained by keeping only sites within an outer radius $R_\text{out}$ of a chosen center, with any hopping whose endpoints leave the flake simply omitted rather than wrapped. The lattice termination acts as a physical hard wall, and the bulk gap that closes and reopens across a topological transition can host low-energy in-gap boundary states.
- **Soft-confinement dot.** An effective dot embedded in a larger flake by adding a smooth radial onsite potential $V(r) = \tfrac{V_0}{2}\left[1 + \tanh\!\left(\tfrac{r - R_\text{dot}}{\xi}\right)\right]$, where $R_\text{dot}$ sets the dot radius, $V_0$ the barrier height, and $\xi$ the softness of the wall. This geometry has two boundaries — the physical outer edge and the internal confinement wall — and adds $V_0$ and $\xi$ as sweep parameters.

Open boundaries also expose **spatial observables** that a PBC phase diagram cannot. For a normalized eigenstate $|\psi_k\rangle$, the site probability density is $\rho_i^{(k)} = |\langle i | \psi_k\rangle|^2$. Summing this density over a region — an outer edge shell, the dot core, or the internal boundary — gives a **participation** value (e.g. edge participation $P_\text{edge}^{(k)}$) that measures how strongly a state localizes there. For selected states, QBP also evaluates the **bond current** $J_{i\to j}^{(k)} = -2\,\mathrm{Im}\!\left[(\psi_i^{(k)})^* H_{ij}\,\psi_j^{(k)}\right]$ (with $\hbar = 1$), which reveals whether a boundary-localized state carries the organized circulation of a chiral edge mode. A phase diagram can then be judged not only by bulk quantities like the spectral gap or ground-state energy, but by whether low-energy states localize at the expected physical or confinement-defined boundaries.

## The Hubbard Model

The **Hubbard model** lives on the same honeycomb lattice with two sublattices $A$ and $B$, but adds two ingredients the Haldane model lacks: an onsite interaction $U$ and an electron spin $\sigma \in \{\uparrow, \downarrow\}$. Its Hamiltonian is

$$
H = -t \sum_{\langle i,j\rangle,\sigma} c_{i,\sigma}^\dagger c_{j,\sigma}
  + U \sum_i c_{i,\uparrow}^\dagger c_{i,\uparrow} c_{i,\downarrow}^\dagger c_{i,\downarrow},
$$

where $t$ is the nearest-neighbor hopping (analogous to the Haldane $t_1$) and $U$ is the onsite Coulomb repulsion.

The Hubbard model's **magnetic phases** are of particular interest: varying $U$, $t$, and $N_\text{occ}$ moves the ground state between paramagnetic, ferromagnetic, and antiferromagnetic order. These are identified by the staggered magnetization $M_\text{stag} = M_A - M_B$ and the total magnetization $M_\text{tot} = M_A + M_B$ — both zero in the paramagnetic phase, only $M_\text{stag}$ zero in the ferromagnetic phase, and only $M_\text{tot}$ zero in the antiferromagnetic phase.

Crucially, the interaction term means the energy can no longer be obtained by reducing to a single-particle Hamiltonian. The full many-body Hamiltonian — exponential in the number of sites, and hence in the number of qubits — must be used instead. This exponential cost is exactly what makes the Hubbard model a compelling target for demonstrating quantum advantage.

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
: A protocol that rewrites fermionic operators as Pauli operators so the Hamiltonian can run on a quantum computer. QBP uses mappings such as [Jordan-Wigner](https://en.wikipedia.org/wiki/Jordan%E2%80%93Wigner_transformation) that send one spin-orbital to exactly one qubit, so the qubit count equals the number of sites.

## Simulation Techniques

QBP supports two quantum algorithms for estimating ground-state energies, plus a classical tensor-network benchmark.

**Variational Quantum Eigensolver (VQE)**
: A hybrid quantum-classical algorithm that exploits the variational principle $E_0 \leq \langle\Psi|\hat{H}|\Psi\rangle / \langle\Psi|\Psi\rangle$. The trial state $|\Psi\rangle = U(\boldsymbol{\theta})|\boldsymbol{0}\rangle$ is prepared by a parameterized circuit (the **ansatz**), and a classical optimizer iteratively updates $\boldsymbol{\theta}$ to minimize the measured energy. Each iteration uses a shallow circuit, which makes VQE well suited to NISQ-era hardware. The user chooses the qubit mapping, the ansatz (e.g. an excitation-preserving ansatz of two-qubit gates), and the classical optimizer (e.g. Adam or SPSA).

**Iterative Quantum Phase Estimation (IQPE)**
: A variant of [quantum phase estimation](https://en.wikipedia.org/wiki/Quantum_phase_estimation_algorithm) that trades qubit count for iteration count. Phase estimation extracts $\theta$ from $U|\psi\rangle = e^{2\pi i\theta}|\psi\rangle$; precision in IQPE is controlled by the number of iterations $m$, which determines the bits $\phi = 0.x_1 x_2 \cdots x_m$ of the phase. Because $\hat{H}$ is not unitary, IQPE simulates the time-evolution operator $U(t) = e^{-i\hat{H}t}$, decomposed into gates via **Trotterization**. The user chooses the evolution time $t$ (typically so that $E_\text{max} t < 2\pi$), the number of Trotter steps $N_\text{trot}$, and the number of iterations $m$. IQPE circuits are deeper than VQE circuits, but when they are viable they are typically more accurate.

**Density Matrix Renormalization Group (DMRG)**
: A classical tensor-network algorithm that serves as a competitive benchmark against the quantum methods. Like VQE and IQPE, DMRG is variational: it minimizes the energy within a restricted family of trial states, representing $|\Psi\rangle$ as a **matrix product state (MPS)** and $\hat{H}$ as a **matrix product operator (MPO)**, each decomposed into local tensors joined by internal *bond* indices. The **bond dimension** caps the entanglement the MPS can represent and is the main accuracy/cost knob. DMRG optimizes the state in **sweeps**, updating neighboring site-tensor pairs against an effective environment Hamiltonian and using a singular value decomposition to discard singular values below a **truncation cutoff**. The user chooses the maximum number of sweeps, the maximum bond dimension, the truncation cutoff, and the initial MPS. QBP runs DMRG through a bundled Julia + ITensorMPS toolchain.

**Analytic, simulated, and noisy runs**
: Every QBP workflow can be executed in three modes. *Analytic* runs diagonalize the Hamiltonian classically and return the exact answer; they are the fastest and serve as a ground-truth reference. *Simulated ideal* runs execute the quantum circuit on a noiseless statevector simulator, so they capture algorithmic error (e.g. Trotter or ansatz error) without hardware noise. *Simulated noisy* runs use a noise model (or a real backend via Qiskit IBM Runtime) and additionally capture device noise.