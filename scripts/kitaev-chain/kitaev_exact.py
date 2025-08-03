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

##params##
n_modes = 6
tunneling = -1.0
superconducting = 1.0
mus = np.linspace(0.0, 3.0, 50)

##collect only ε₀ and ε₁##
epsilon_0 = []
epsilon_1 = []

for mu in mus:
    H = kitaev_hamiltonian(n_modes, tunneling, superconducting, mu)
    _, eps, _ = H.diagonalizing_bogoliubov_transform()
    eps_sorted = np.sort(np.real(eps))
    epsilon_0.append(eps_sorted[0])
    epsilon_1.append(eps_sorted[1])
print(epsilon_0)
print(epsilon_1)

epsilon_0 = np.array(epsilon_0)
epsilon_1 = np.array(epsilon_1)

##plot only ±ε₀ and ±ε₁##
plt.figure(figsize=(7, 5))

# Central zero modes
plt.plot(mus, epsilon_0, color="black", label=r"$\epsilon_0$")
plt.plot(mus, -epsilon_0, color="black")

# First excited modes
plt.plot(mus, epsilon_1, color="gray", linestyle="--", label=r"$\epsilon_1$")
plt.plot(mus, -epsilon_1, color="gray", linestyle="--")

plt.axhline(0, color='black', linewidth=0.5, linestyle=':')
plt.xlabel("Chemical Potential (μ)")
plt.ylabel("Excitation Energy")
plt.title("Ideal BdG Spectrum – Central 4 Modes (n = 6)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
