from utils import orbital_combinations, data_exact
import numpy as np
import matplotlib.pyplot as plt
import os

n_modes = 7
tunneling = -1.0
superconducting = 1.0
chemical_potential_values = list(np.linspace(0.0, 3.0, num=50))
occupied_orbitals_list = list(orbital_combinations(n_modes, threshold=2))

data = data_exact(
    n_modes=n_modes,
    tunneling=tunneling,
    superconducting=superconducting,
    chemical_potential_values=chemical_potential_values,
    occupied_orbitals_list=occupied_orbitals_list,
)

color = ['orange', 'green']
orbitals = [(0,), (1,)]
for i in range(2*len(orbitals)):
    plt.plot(chemical_potential_values, data['bdg_energy_exact'][0][i], color=color[i%2], label=str(orbitals[i] if i < 2 else ""))

plt.title("Ideal BdG Spectrum for 1e and 2e ("+str(n_modes)+" modes)")
plt.legend(fontsize='small', loc='right')

file_path = os.path.join(os.getcwd(), "..", "..", "plots/kitaev-chain/"+str(n_modes)+"-modes/bdg-energy-analytic.png")
plt.savefig(file_path)