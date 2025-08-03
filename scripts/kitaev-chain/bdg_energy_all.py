from utils import orbital_combinations, data_exact, data_simulated
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke

n_modes = 7
tunneling = -1.0
superconducting = 1.0
chemical_potential_values = list(np.linspace(0.0, 3.0, num=10))
occupied_orbitals_list = list(orbital_combinations(n_modes, threshold=2))
backend = FakeSherbrooke()
shots = 100000

data = {
    'exact': data_exact(
        n_modes=n_modes,
        tunneling=tunneling,
        superconducting=superconducting,
        chemical_potential_values=chemical_potential_values,
        occupied_orbitals_list=occupied_orbitals_list,
    ),
    'simulated_ideal': data_simulated(
        n_modes=n_modes,
        tunneling=tunneling,
        superconducting=superconducting,
        chemical_potential_values=chemical_potential_values,
        occupied_orbitals_list=occupied_orbitals_list,
        mitigation=False,
    ),
    'simulated_unmitigated': data_simulated(
        n_modes=n_modes,
        tunneling=tunneling,
        superconducting=superconducting,
        chemical_potential_values=chemical_potential_values,
        occupied_orbitals_list=occupied_orbitals_list,
        backend=backend,
        mitigation=False,
    ),
    'simulated_mitigated': data_simulated(
        n_modes=n_modes,
        tunneling=tunneling,
        superconducting=superconducting,
        chemical_potential_values=chemical_potential_values,
        occupied_orbitals_list=occupied_orbitals_list,
        backend=backend,
        mitigation=True,
    )
}

idx = 0
for val in data.values():
    color = ['orange', 'green', 'blue', 'red']
    orbitals = [(0,), (1,)]
    for i in range(2*len(orbitals)):
        plt.plot(chemical_potential_values, val['bdg_energy_exact' if idx == 0 else 'bdg_energy_simulated'][0][i], '-' if idx == 0 else '.', color=color[idx], label=str(orbitals[i] if i < 2 else ""))
    idx += 1

legend_elements = [
    Line2D([0], [0], color='orange', lw=2, label='Exact Values'),
    Line2D([0], [0], marker='o', color='w', label='Ideal Simulated', markerfacecolor='green', markersize=5),
    Line2D([0], [0], marker='o', color='w', label='Noisy Unmitigated', markerfacecolor='blue', markersize=5),
    Line2D([0], [0], marker='o', color='w', label='Noisy Mitigated', markerfacecolor='red', markersize=5)
]

plt.title("BdG Spectrum for 1e and 2e ("+str(n_modes)+" modes, "+str(shots)+" shots)")
plt.legend(handles=legend_elements, fontsize='x-small', loc='right')

file_path = os.path.join(os.getcwd(), "..", "..", "plots/kitaev-chain/"+str(n_modes)+"-modes/bdg-energy-all.png")
plt.savefig(file_path)