import os
import time
import numpy as np
import matplotlib.pyplot as plt
import ujson
from collections import defaultdict
from joblib import Parallel, delayed
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke

from utils_mitigated import (
    error_mitigation,
    orbital_combinations,
    get_exact_energies,
)

N_MODES = 7
TUNNELING = -1.0
SUPERCONDUCTING = 1.0
CHEMICAL_POTENTIALS = np.linspace(0.0, 3.0, 10)
OCCUPIED_ORBITALS = list(orbital_combinations(N_MODES, threshold=2))
BACKEND = FakeSherbrooke()
SHOTS = 100000
EXECUTE = True


def main():
    cache_dir = os.path.join(os.getcwd(), '../logs/kitaev-chain')
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f'data_{N_MODES}_modes_mitigated.json')

    if EXECUTE:
        tasks = [
            (mu, oo) for mu in CHEMICAL_POTENTIALS for oo in OCCUPIED_ORBITALS
        ]

        start_time = time.time()
        results_list = Parallel(n_jobs=-1, verbose=10)(
            delayed(error_mitigation)(
                n_modes=N_MODES,
                tunneling=TUNNELING,
                superconducting=SUPERCONDUCTING,
                chemical_potential=mu,
                occupied_orbitals=oo,
                backend=BACKEND,
                shots=SHOTS,
            ) for mu, oo in tasks
        )

        processed_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for mu, oo, results_dict in results_list:
            for stage, data in results_dict.items():
                processed_results[stage][str(oo)][str(mu)] = data

        with open(cache_file, 'w') as f:
            ujson.dump(processed_results, f)

        pipeline_data = processed_results
    else:
        with open(cache_file, 'r') as f:
            pipeline_data = ujson.load(f)

    plots_dir = os.path.join(os.getcwd(), "../plots", f"{N_MODES}-modes")
    os.makedirs(plots_dir, exist_ok=True)

    print("INFO: Generating error mitigation fidelity plot...")
    plot_error_mitigation(pipeline_data, plots_dir)

    print("INFO: Generating BdG energy plot...")
    plot_bdg_energy(pipeline_data, plots_dir)


def plot_error_mitigation(pipeline_data, plots_dir):
    fig, ax = plt.subplots(figsize=(10, 7), dpi=150)

    mit_stages = ['raw', 'mem', 'ps', 'pur']
    labels = ['Raw', '+MEM', '+PS', '+Pur.']
    markers = ['o', '^', 'D', 's']
    colors = ['C0', 'C1', 'C4', 'C3']

    for i, stage in enumerate(mit_stages):
        avg_fidelity = []
        avg_stddev = []
        for mu_str in [str(mu) for mu in CHEMICAL_POTENTIALS]:
            fid_vals_for_mu = []
            std_vals_for_mu = []
            for oo_str in pipeline_data[stage]:
                data = pipeline_data[stage][oo_str].get(mu_str)
                if data:
                    fid_vals_for_mu.append(data['fidelity_witness'][0])
                    std_vals_for_mu.append(data['fidelity_witness'][1])

            if fid_vals_for_mu:
                avg_fidelity.append(np.mean(fid_vals_for_mu))
                avg_stddev.append(np.sqrt(np.sum(np.array(std_vals_for_mu) ** 2)) / len(std_vals_for_mu))

        ax.errorbar(
            CHEMICAL_POTENTIALS, 1 - np.array(avg_fidelity), yerr=2 * np.array(avg_stddev),
            fmt=f'{markers[i]}:', color=colors[i], label=labels[i], capsize=4, markersize=8,
            linestyle='dotted'
        )

    ax.set_title(f"Fidelity Improvement at Each Mitigation Step ({N_MODES} modes)", fontsize=16)
    ax.set_xlabel("Chemical Potential (μ)", fontsize=14)
    ax.set_ylabel(r"$1 - F_W$ (Error)", fontsize=14)
    ax.set_yscale('log')
    ax.legend(fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    file_path = os.path.join(plots_dir, "fidelity.png")
    plt.savefig(file_path, bbox_inches='tight')
    plt.close(fig)
    print(f"SUCCESS: Saved fidelity plot to {file_path}")


def plot_bdg_energy(pipeline_data, plots_dir):
    fig, ax = plt.subplots(figsize=(10, 7), dpi=150)

    exact_energies = get_exact_energies(N_MODES, TUNNELING, SUPERCONDUCTING, CHEMICAL_POTENTIALS, OCCUPIED_ORBITALS)
    low_exact = exact_energies[()]
    high_exact = exact_energies[tuple(range(N_MODES))]
    combs = list(orbital_combinations(N_MODES, threshold=2))

    for i in range(2):
        particle_exact = exact_energies[combs[2 * i + 2]]
        hole_exact = exact_energies[combs[2 * i + 3]]
        ax.plot(CHEMICAL_POTENTIALS, particle_exact - low_exact, '-', color='black', alpha=0.8)
        ax.plot(CHEMICAL_POTENTIALS, hole_exact - high_exact, '-', color='black', alpha=0.8,
                label='Exact' if i == 0 else "")

    def get_bdg_from_results(stage_data):
        bdg_energy = np.zeros((4, len(CHEMICAL_POTENTIALS)))

        low_data = stage_data[str(())]
        high_data = stage_data[str(tuple(range(N_MODES)))]
        low_vals = np.array([low_data[str(mu)]['energy'][0] for mu in CHEMICAL_POTENTIALS])
        high_vals = np.array([high_data[str(mu)]['energy'][0] for mu in CHEMICAL_POTENTIALS])

        for i in range(2):
            p_oo_str = str(combs[2 * i + 2])
            h_oo_str = str(combs[2 * i + 3])
            particle_vals = np.array([stage_data[p_oo_str][str(mu)]['energy'][0] for mu in CHEMICAL_POTENTIALS])
            hole_vals = np.array([stage_data[h_oo_str][str(mu)]['energy'][0] for mu in CHEMICAL_POTENTIALS])
            bdg_energy[i] = particle_vals - low_vals
            bdg_energy[2 + i] = hole_vals - high_vals
        return bdg_energy

    bdg_raw = get_bdg_from_results(pipeline_data['raw'])
    for i in range(4):
        ax.plot(CHEMICAL_POTENTIALS, bdg_raw[i], 'o', color='C0', markersize=4,
                label='Unmitigated (Raw)' if i == 0 else "")

    bdg_pur = get_bdg_from_results(pipeline_data['pur'])
    for i in range(4):
        ax.plot(CHEMICAL_POTENTIALS, bdg_pur[i], 'o', color='C3', markersize=4,
                label='Fully Mitigated (+Pur)' if i == 0 else "")

    ax.set_title(f"BdG Spectrum ({N_MODES} modes, {SHOTS} shots)", fontsize=16)
    ax.set_xlabel("Chemical Potential (μ)", fontsize=14)
    ax.set_ylabel("Excitation Energy", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    file_path = os.path.join(plots_dir, "bdg-energy-mitigated.png")
    plt.savefig(file_path, bbox_inches='tight')
    plt.close(fig)
    print(f"SUCCESS: Saved BdG spectrum plot to {file_path}")


if __name__ == "__main__":
    main()