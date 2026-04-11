import os
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
import multiprocessing

from utils_parallelized import orbital_combinations, data_exact, data_simulated

N_MODES = 7
TUNNELING = -1.0
SUPERCONDUCTING = 1.0
CHEMICAL_POTENTIAL_VALUES = list(np.linspace(0.0, 3.0, num=10))
OCCUPIED_ORBITALS_LIST = list(orbital_combinations(N_MODES, threshold=2))
BACKEND = FakeSherbrooke()
SHOTS = 100000
EXECUTE = True


def main():
    start_time = time.time()
    print(f"INFO: Running on {multiprocessing.cpu_count()} cores.")

    cache_file = os.path.join(os.getcwd(), '..', 'logs', 'kitaev-chain', f'data_parallelized_{N_MODES}_modes.pkl')
    if EXECUTE:
        data = {
            'exact': data_exact(
                n_modes=N_MODES,
                tunneling=TUNNELING,
                superconducting=SUPERCONDUCTING,
                chemical_potential_values=CHEMICAL_POTENTIAL_VALUES,
                occupied_orbitals_list=OCCUPIED_ORBITALS_LIST,
            ),
            'simulated_ideal': data_simulated(
                n_modes=N_MODES,
                tunneling=TUNNELING,
                superconducting=SUPERCONDUCTING,
                chemical_potential_values=CHEMICAL_POTENTIAL_VALUES,
                occupied_orbitals_list=OCCUPIED_ORBITALS_LIST,
                backend=None,
                mitigation=False,
                shots=SHOTS,
            ),
            'simulated_unmitigated': data_simulated(
                n_modes=N_MODES,
                tunneling=TUNNELING,
                superconducting=SUPERCONDUCTING,
                chemical_potential_values=CHEMICAL_POTENTIAL_VALUES,
                occupied_orbitals_list=OCCUPIED_ORBITALS_LIST,
                backend=BACKEND,
                mitigation=False,
                shots=SHOTS,
            ),
            'simulated_mitigated': data_simulated(
                n_modes=N_MODES,
                tunneling=TUNNELING,
                superconducting=SUPERCONDUCTING,
                chemical_potential_values=CHEMICAL_POTENTIAL_VALUES,
                occupied_orbitals_list=OCCUPIED_ORBITALS_LIST,
                backend=BACKEND,
                mitigation=True,
                shots=SHOTS,
            )
        }
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, "wb") as f:
            pickle.dump(data, f)
        print(f"INFO: Data generated and saved to {cache_file}")
    else:
        print(f"INFO: Loading data from {cache_file}")
        with open(cache_file, "rb") as f:
            data = pickle.load(f)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['orange', 'green', 'blue', 'red']
    labels = ['Exact', 'Ideal Simulated', 'Noisy Unmitigated', 'Noisy Mitigated']
    styles = ['-', '.', '.', '.']

    for i, (key, val) in enumerate(data.items()):
        data_key = 'bdg_energy_exact' if key == 'exact' else 'bdg_energy_simulated'
        if data_key not in val:
            print(f"WARN: Missing '{data_key}' in data for '{key}'. Skipping plot.")
            continue

        bdg_energies, _ = val[data_key]
        num_bdg_plots = bdg_energies.shape[0]

        for j in range(num_bdg_plots):
            plot_label = labels[i] if j == 0 else ""
            ax.plot(CHEMICAL_POTENTIAL_VALUES, bdg_energies[j], styles[i], color=colors[i], label=plot_label)

    legend_elements = [
        Line2D([0], [0], color='orange', lw=2, label='Exact Values'),
        Line2D([0], [0], marker='o', color='w', label='Ideal Simulated', markerfacecolor='green', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Noisy Unmitigated', markerfacecolor='blue', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Noisy Mitigated', markerfacecolor='red', markersize=8)
    ]

    ax.set_title(f"BdG Spectrum ({N_MODES} modes, {shots_str(SHOTS)} shots)")
    ax.set_xlabel("Chemical Potential (μ)")
    ax.set_ylabel("Excitation Energy")
    ax.legend(handles=legend_elements)
    ax.grid(True, linestyle='--', alpha=0.6)

    plots_dir = os.path.join(os.getcwd(), "..", "plots", f"{N_MODES}-modes")
    os.makedirs(plots_dir, exist_ok=True)
    file_path = os.path.join(plots_dir, "bdg-energy-all.png")
    plt.savefig(file_path)
    print(f"SUCCESS: Plot saved to {file_path}")

    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")


def shots_str(shots):
    if shots >= 1000:
        return f"{shots // 1000}k"
    return str(shots)


if __name__ == "__main__":
    main()