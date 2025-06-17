from utils import orbital_combinations, circuit_depth
import numpy as np
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke

n_modes = 7
tunneling = -1.0
superconducting = 1.0
chemical_potential_values = list(np.linspace(0.0, 3.0, num=10))
occupied_orbitals_list = list(orbital_combinations(n_modes, threshold=2))
backend = FakeSherbrooke()

data = {
    'exact': circuit_depth(
        n_modes=n_modes,
        tunneling=tunneling,
        superconducting=superconducting,
        chemical_potential_values=chemical_potential_values,
        occupied_orbitals_list=occupied_orbitals_list,
    ),
    'simulated_ideal': circuit_depth(
        n_modes=n_modes,
        tunneling=tunneling,
        superconducting=superconducting,
        chemical_potential_values=chemical_potential_values,
        occupied_orbitals_list=occupied_orbitals_list,
    ),
    'simulated_noisy': circuit_depth(
        n_modes=n_modes,
        tunneling=tunneling,
        superconducting=superconducting,
        chemical_potential_values=chemical_potential_values,
        occupied_orbitals_list=occupied_orbitals_list,
        backend=backend,
    )
}

for key, value in data.items():
    data[key] = sum(value.values())/len(value.values())

print(data)