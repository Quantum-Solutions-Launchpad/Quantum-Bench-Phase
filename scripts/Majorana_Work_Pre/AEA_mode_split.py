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
mus = np.linspace(0.0, 3.0, 100)
n_values = list(range(4, 30))  ##you can extend this range##

epsilon_0_max_deviation = []

for n in n_values:
    eps_0_mu = []
    for mu in mus:
        H = kitaev_hamiltonian(n, t, delta, mu)
        _, eps, _ = H.diagonalizing_bogoliubov_transform()
        eps_sorted = np.sort(np.real(eps))
        eps_0_mu.append(abs(eps_sorted[0]))  ##absolute deviation from zero##

    epsilon_0_max_deviation.append(max(eps_0_mu))

##plotting##
plt.figure(figsize=(8, 5))
plt.plot(n_values, epsilon_0_max_deviation, marker='o')
plt.yscale('log')  ##Majorana splitting should decay exponentially!##
plt.xlabel("Chain Length (n)")
plt.ylabel(r"max$_{\mu}$|$\epsilon_0$|")
plt.title("Majorana Mode Splitting vs Chain Length")
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig("epsilon0_vs_n.png")
plt.show()
