import os
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
from joblib import Parallel, delayed

# Import the fully-featured and refactored utility functions
from utils_test import (
    sub_data_simulated,
    orbital_combinations,
    data_exact,
    compute_correlation_matrix,
    post_select_quasis,
    purify_idempotent_matrix,
    fidelity_witness,
    kitaev_hamiltonian,
    diagonalizing_bogoliubov_transform,
    expectation_from_correlation_matrix,
)

# --- Experiment Parameters ---
N_MODES = 7
TUNNELING = -1.0
SUPERCONDUCTING = 1.0
CHEMICAL_POTENTIAL_VALUES = list(np.linspace(0.0, 3.0, num=10))
OCCUPIED_ORBITALS_LIST = list(orbital_combinations(N_MODES, threshold=2))
BACKEND = FakeSherbrooke()
SHOTS = 100000
EXECUTE = True


# --- Helper function for pickling ---
def nested_dict_factory():
    """A named function to replace the lambda for defaultdict, making it picklable."""
    return defaultdict(dict)


def _run_analysis(params):
    """
    This function mirrors the structure of IBM's `_run_analysis`.
    It gets all the raw data via parallel simulation, then sequentially applies
    all mitigation and analysis steps.
    """
    n_modes = params['n_modes']

    # --- I. Parallel Data Acquisition (unchanged) ---
    print("===== Starting Parallel Data Acquisition =====")
    simulation_tasks = [
        {'chemical_potential': mu, 'occupied_orbitals': oo, **params}
        for mu in params['chemical_potential_values']
        for oo in params['occupied_orbitals_list']
    ]
    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(sub_data_simulated)(p) for p in simulation_tasks
    )
    all_quasis = defaultdict(dict)
    for mu, oo, quasis in results:
        all_quasis[mu][oo] = quasis
    print("\n===== Data Acquisition Complete. Starting Mitigation & Analysis Pipeline =====")

    # --- II. Sequential Mitigation & Analysis Pipeline ---
    corr_matrices = {}
    quasi_dists = {}

    for mu in params['chemical_potential_values']:
        trans_mat_exact, _, _, hamiltonian_parity = diagonalizing_bogoliubov_transform(
            n_modes, params['tunneling'], params['superconducting'], mu
        )
        for oo in params['occupied_orbitals_list']:
            quasis_mem = all_quasis[mu][oo]
            exact_parity = (-1) ** len(oo) * hamiltonian_parity

            quasis_ps, _ = post_select_quasis(
                quasis_mem, n_modes=n_modes, exact_parity=exact_parity
            )

            # Store the quasis distributions
            quasi_dists[(mu, oo, 'raw')] = quasis_mem  # 'raw' and 'mem' are the same in this flow
            quasi_dists[(mu, oo, 'mem')] = quasis_mem
            quasi_dists[(mu, oo, 'ps')] = quasis_ps

            # Compute correlation matrices for pre-purification stages
            corr_raw, cov_raw = compute_correlation_matrix(quasis_mem)
            corr_ps, cov_ps = compute_correlation_matrix(quasis_ps)

            corr_matrices[(mu, oo, 'raw')] = (corr_raw, cov_raw)
            corr_matrices[(mu, oo, 'mem')] = (corr_raw, cov_raw)  # Same as raw
            corr_matrices[(mu, oo, 'ps')] = (corr_ps, cov_ps)

            # --- THIS IS THE FIX ---
            # Purify ONLY if the post-selected matrix is valid (not all zeros).
            # A valid correlation matrix has a trace equal to the number of modes.
            # A zero matrix will have a trace of zero.
            if not np.allclose(corr_ps, 0):
                corr_pur = purify_idempotent_matrix(corr_ps)
                # Only add the 'pur' entry if purification was successful
                corr_matrices[(mu, oo, 'pur')] = (corr_pur, cov_ps)
            # If the matrix was all zeros, we simply don't add a 'pur' entry for this
            # (mu, oo) data point. It will be gracefully skipped later.
            # --- END OF FIX ---

    # --- III. Compute Final Observables (unchanged, but now safer) ---
    print("===== Analysis Pipeline Complete. Calculating Final Observables. =====")
    analysis_results = defaultdict(nested_dict_factory)
    mit_stages = ['raw', 'mem', 'ps', 'pur']

    for mu in params['chemical_potential_values']:
        # ... (hamiltonian and exact correlation matrix calculation is correct) ...
        trans_mat_exact, _, _, _ = diagonalizing_bogoliubov_transform(
            n_modes, params['tunneling'], params['superconducting'], mu
        )
        hamiltonian_quad = kitaev_hamiltonian(n_modes, params['tunneling'], params['superconducting'], mu)
        for oo in params['occupied_orbitals_list']:
            occupation = np.zeros(n_modes)
            occupation[list(oo)] = 1.0
            W1, W2 = trans_mat_exact[:, :n_modes], trans_mat_exact[:, n_modes:]
            full_trans_mat = np.block([[W1, W2], [W2.conj(), W1.conj()]])
            corr_diag = np.diag(np.concatenate([occupation, 1 - occupation]))
            corr_exact = full_trans_mat.T.conj() @ corr_diag @ full_trans_mat

            for stage in mit_stages:
                # Use .get() to safely handle potentially missing 'pur' keys
                corr, cov = corr_matrices.get((mu, oo, stage), (None, None))
                if corr is None:
                    continue  # Skip this stage if data doesn't exist

                energy, E_std = np.real(expectation_from_correlation_matrix(hamiltonian_quad, corr, cov))
                fid, fid_std = fidelity_witness(corr, corr_exact, cov)
                analysis_results[stage][oo][mu] = {'energy': (energy, E_std), 'fidelity': (fid, fid_std)}

    return analysis_results


def main():
    start_time = time.time()
    params = {
        'n_modes': N_MODES,
        'tunneling': TUNNELING,
        'superconducting': SUPERCONDUCTING,
        'chemical_potential_values': CHEMICAL_POTENTIAL_VALUES,
        'occupied_orbitals_list': OCCUPIED_ORBITALS_LIST,
        'backend': BACKEND,
        'mitigation': True,
        'shots': SHOTS,
    }

    cache_file = os.path.join(os.getcwd(), '../cache', f'mitigated_data_{N_MODES}_modes.pkl')
    if EXECUTE:
        analysis_results = _run_analysis(params)
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(analysis_results, f)
    else:
        print(f"INFO: Loading cached results from {cache_file}...")
        with open(cache_file, 'rb') as f:
            analysis_results = pickle.load(f)

    analysis_results = dict(analysis_results)

    # --- Plotting ---
    plots_dir = os.path.join(os.getcwd(), "../plots", f"{N_MODES}-modes")
    os.makedirs(plots_dir, exist_ok=True)

    print("INFO: Generating error mitigation fidelity plot...")
    plot_fidelity_witness(analysis_results, params, plots_dir)

    # --- NEW: Call the BdG energy plot function ---
    print("INFO: Generating BdG energy spectrum plot...")
    plot_bdg_energy(analysis_results, params, plots_dir)

    print(f"Total time: {time.time() - start_time:.2f}s")


# --- Plotting function for Fidelity is unchanged ---
def plot_fidelity_witness(analysis_results, params, plots_dir):
    # ... (This function is correct and does not need changes) ...
    fig, ax = plt.subplots(figsize=(10, 7))
    mit_stages = ['raw', 'mem', 'ps', 'pur']
    labels = ['Raw', '+MEM', '+PS', '+Pur.']
    markers = ['o', '^', 'D', 's']
    for i, stage in enumerate(mit_stages):
        avg_fidelity = []
        avg_stddev = []
        for mu in params['chemical_potential_values']:
            fid_vals = [analysis_results[stage][oo][mu]['fidelity'][0] for oo in params['occupied_orbitals_list']]
            std_vals = [analysis_results[stage][oo][mu]['fidelity'][1] for oo in params['occupied_orbitals_list']]
            avg_fidelity.append(np.mean(fid_vals))
            avg_stddev.append(np.sqrt(np.sum(np.array(std_vals) ** 2)) / len(fid_vals) if fid_vals else 0)
        ax.errorbar(
            params['chemical_potential_values'], 1 - np.array(avg_fidelity), yerr=2 * np.array(avg_stddev),
            fmt=f'{markers[i]}:', label=labels[i], capsize=4
        )
    ax.set_title(f"Fidelity Improvement at Each Mitigation Step ({params['n_modes']} modes)")
    ax.set_xlabel("Chemical Potential (μ)")
    ax.set_ylabel(r"$1 - F_W$ (Error)")
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, which='both', linestyle='--')
    file_path = os.path.join(plots_dir, "fidelity.png")
    plt.savefig(file_path, bbox_inches='tight')
    plt.close(fig)
    print(f"SUCCESS: Saved fidelity plot to {file_path}")


# --- NEW: Plotting function for BdG Energy Spectrum ---
def plot_bdg_energy(analysis_results, params, plots_dir):
    """Plots the BdG spectrum comparing raw data vs fully mitigated data."""
    fig, ax = plt.subplots(figsize=(10, 7), dpi=150)

    n_modes = params['n_modes']
    chemical_potentials = params['chemical_potential_values']

    # --- 1. Get Exact Data for the "Ideal" lines ---
    exact_data = data_exact(
        n_modes=n_modes,
        tunneling=params['tunneling'],
        superconducting=params['superconducting'],
        chemical_potential_values=chemical_potentials,
        occupied_orbitals_list=params['occupied_orbitals_list']
    )
    bdg_exact, _ = exact_data['bdg_energy_exact']

    # Plot the exact lines
    for i in range(bdg_exact.shape[0]):
        ax.plot(chemical_potentials, bdg_exact[i], '-', color='black', alpha=0.8, label='Exact' if i == 0 else "")

    # --- 2. Process and Plot Simulated Data ---

    # Helper to extract BdG energies from the nested results dictionary
    def get_bdg_from_results(stage_data):
        # Determine the number of particle/hole pairs from the orbital list
        threshold = 0
        combs = list(orbital_combinations(n_modes))
        for i in range(0, len(combs), 2):
            if (combs[i] in OCCUPIED_ORBITALS_LIST and combs[i + 1] in OCCUPIED_ORBITALS_LIST):
                threshold += 1
            else:
                break
        threshold -= 1  # Adjust because first pair is ground states

        bdg_energy = np.zeros((2 * threshold, len(chemical_potentials)))

        # Get ground state energies
        low_oo, high_oo = (), tuple(range(n_modes))
        low_vals = np.array([stage_data[low_oo][mu]['energy'][0] for mu in chemical_potentials])
        high_vals = np.array([stage_data[high_oo][mu]['energy'][0] for mu in chemical_potentials])

        # Calculate excitation energies
        for i in range(threshold):
            p_oo = combs[2 * (i + 1)]  # Particle state
            h_oo = combs[2 * (i + 1) + 1]  # Hole state

            particle_vals = np.array([stage_data[p_oo][mu]['energy'][0] for mu in chemical_potentials])
            hole_vals = np.array([stage_data[h_oo][mu]['energy'][0] for mu in chemical_potentials])

            bdg_energy[i] = particle_vals - low_vals
            bdg_energy[threshold + i] = hole_vals - high_vals

        return bdg_energy

    # Get BdG for raw (unmitigated) data
    bdg_raw = get_bdg_from_results(analysis_results['raw'])
    for i in range(bdg_raw.shape[0]):
        ax.plot(chemical_potentials, bdg_raw[i], 'o', color='C0', markersize=5,
                label='Raw' if i == 0 else "")

    # Get BdG for fully mitigated data
    bdg_pur = get_bdg_from_results(analysis_results['pur'])
    for i in range(bdg_pur.shape[0]):
        ax.plot(chemical_potentials, bdg_pur[i], 's', color='C3', markersize=5,
                label='Mitigated (+Pur)' if i == 0 else "")

    # --- 3. Finalize and Save Plot ---
    ax.set_title(f"BdG Spectrum ({n_modes} modes, {params['shots'] // 1000}k shots)")
    ax.set_xlabel("Chemical Potential (μ)")
    ax.set_ylabel("Excitation Energy")
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    file_path = os.path.join(plots_dir, "bdg-energy-mitigated.png")
    plt.savefig(file_path, bbox_inches='tight')
    plt.close(fig)
    print(f"SUCCESS: Saved BdG spectrum plot to {file_path}")


if __name__ == "__main__":
    main()