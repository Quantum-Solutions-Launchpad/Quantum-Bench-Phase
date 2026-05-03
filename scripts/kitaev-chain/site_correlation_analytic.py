from utils import orbital_combinations, data_exact
import matplotlib.pyplot as plt
import os

n_modes = 7
tunneling = -1.0
superconducting = 1.0
chemical_potential_values = [1.5]
occupied_orbitals_list = list(orbital_combinations(n_modes, threshold=2))

data = data_exact(
    n_modes=n_modes,
    tunneling=tunneling,
    superconducting=superconducting,
    chemical_potential_values=chemical_potential_values,
    occupied_orbitals_list=occupied_orbitals_list,
)

color = ['orange', 'green']
keys = list(set(i for i in data['site_correlation_exact'].keys() if i[1] == () or i[1] == (0,)))
for i in range(len(keys)):
    plt.plot(data['site_correlation_exact'][keys[i]], color=color[i], label=str(keys[i][1]))

plt.title("Ideal Site Correlation for g and 1e ("+str(n_modes)+" modes, μ="+str(chemical_potential_values[0])+")")
plt.legend(fontsize='small', loc='upper left')

file_path = os.path.join(os.getcwd(), "..", "..", "plots/kitaev-chain/"+str(n_modes)+"-modes/site-correlation-exact_mu-"+str(chemical_potential_values[0])+".png")
plt.savefig(file_path)