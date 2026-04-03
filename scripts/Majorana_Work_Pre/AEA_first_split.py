import numpy as np
import matplotlib.pyplot as plt
from qiskit_nature.second_q.hamiltonians import QuadraticHamiltonian

##define Kitaev Hamiltonian##
def kitaev_hamiltonian(n_modes, tunneling, superconducting, chemical_potential):
    eye = np.eye(n_modes)
    upper_diag = np.diag(np.ones(n_modes - 1), k=1)
    lower_diag = np.diag(np.ones(n_modes - 1), k=-1)
    hermitian = -tunneling * (upper_diag + lower_diag) + chemical_potential * eye
    antisymmetric = superconducting * (upper_diag - lower_diag)
    constant = -0.5 * chemical_potential * n_modes
    return QuadraticHamiltonian(
        hermitian_part=hermitian,
        antisymmetric_part=antisymmetric,
        constant=constant
    )

##parameters##
t = -1.0
delta = 1.0
mus = np.linspace(0.0, 3.0, 200)
n_values = list(range(4, 1000))
threshold = 1e-3  ##epsilon_0 splitting threshold

mu_onset = []

for n in n_values:
    found = False
    for mu in mus:
        H = kitaev_hamiltonian(n, t, delta, mu)
        _, eps, _ = H.diagonalizing_bogoliubov_transform()
        eps_sorted = np.sort(np.real(eps))
        if abs(eps_sorted[0]) > threshold:
            mu_onset.append(mu)
            found = True
            break
    if not found:
        mu_onset.append(np.nan)  ## no deviation found

##plotting##
plt.figure(figsize=(8, 5))
plt.plot(n_values, mu_onset, marker='o')
plt.xlabel("Chain Length (n)")
plt.ylabel(r"Onset of $|\epsilon_0| > 10^{-3}$ (Chemical Potential $\mu$)")
plt.title("First Deviation from Zero Mode vs Chain Length")
plt.grid(True)
plt.tight_layout()
plt.savefig("epsilon0_deviation_vs_n.png")
plt.show()
