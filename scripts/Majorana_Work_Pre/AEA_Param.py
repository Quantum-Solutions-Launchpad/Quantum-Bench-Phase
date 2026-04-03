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

##reusable plot function##
def plot_spectra(vary_param, values, fixed_params, filename, title_prefix):
    mus = np.linspace(0.0, 3.0, 50)
    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    axes = axes.flatten()

    for idx, val in enumerate(values):
        params = fixed_params.copy()
        params[vary_param] = val

        epsilon_0 = []
        epsilon_1 = []

        for mu in mus:
            H = kitaev_hamiltonian(params["n_modes"], params["t"], params["delta"], mu)
            _, eps, _ = H.diagonalizing_bogoliubov_transform()
            eps_sorted = np.sort(np.real(eps))
            epsilon_0.append(eps_sorted[0])
            epsilon_1.append(eps_sorted[1])

        ax = axes[idx]
        ax.plot(mus, epsilon_0, color="black", label=r"$\epsilon_0$")
        ax.plot(mus, -np.array(epsilon_0), color="black")
        ax.plot(mus, epsilon_1, color="gray", linestyle="--", label=r"$\epsilon_1$")
        ax.plot(mus, -np.array(epsilon_1), color="gray", linestyle="--")
        ax.axhline(0, color='black', linewidth=0.5, linestyle=':')
        ax.set_title(f"{title_prefix} = {val:.2f}" if isinstance(val, float) else f"{title_prefix} = {val}")
        ax.set_xlabel("μ")
        ax.set_ylabel("Energy")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

##parameter ranges##
t_values = np.linspace(-2.0, -0.1, 10)
delta_values = np.linspace(2.0, 0.1, 10)
n_values = list(range(4, 14))

##run and save##
plot_spectra("t", t_values, {"n_modes": 6, "delta": 1.0}, "t_variation_10.png", "t")
plot_spectra("delta", delta_values, {"n_modes": 6, "t": -1.0}, "delta_variation_10.png", "Δ")
plot_spectra("n_modes", n_values, {"t": -1.0, "delta": 1.0}, "n_modes_variation_10.png", "n")
