from utils_test import band_structure_vqe, band_structure_exact
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
import os
import json
from itertools import product
from joblib import Parallel, delayed
from qiskit_ibm_runtime.fake_provider import FakeManilaV2
import sys
import time

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

n_sites = 4
t1, t2, M = 1.0, 0.05, 0.2

a_vecs = {
    4: [np.array([0.0, -1.0]), np.array([0.0, 1.0]), np.array([1.0, 0.0]), np.array([-1.0, 0.0])],
    6: [np.array([0.0, -1.0]), np.array([np.sqrt(3) / 2, 0.5]), np.array([-np.sqrt(3) / 2, 0.5])],
}.get(n_sites, [])

b_vecs = {
    4: [np.array([-1.0, -1.0]), np.array([1.0, -1.0]), np.array([-1.0, 1.0]), np.array([1.0, 1.0])],
    6: [a_vecs[1] - a_vecs[2], a_vecs[2] - a_vecs[0], a_vecs[0] - a_vecs[1]],
}.get(n_sites, [])

samples = 6

hw_info = FakeManilaV2()
simulator_info = AerSimulator.from_backend(hw_info)
noise_model_info = NoiseModel.from_backend(hw_info)

print(f"  n_sites: {n_sites}")
print(f"  t1={t1}, t2={t2}, M={M}")
print(f"  samples: {samples}x{samples} = {samples ** 2} k-points")
print(f"  schedule_backend: {type(hw_info).__name__}")
print(
    f"  exec_backend: AerSimulator.from_backend({type(hw_info).__name__}) + NoiseModel.from_backend({type(hw_info).__name__})")
print("")

x_list = [float(kx) for kx in np.linspace(-np.pi, np.pi, samples)]
y_list = [float(ky) for ky in np.linspace(-np.pi, np.pi, samples)]
k_points = list(product(x_list, y_list))

try:
    from utils_test import USE_M3, get_thread_local_objects
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp

    if USE_M3:
        backend_objects, m3_cache = get_thread_local_objects()
        simulator = backend_objects['simulator']

        qc = QuantumCircuit(1)
        h = SparsePauliOp(['Z'], [1.0])
        from utils_test import _expectation_1q_m3

        _expectation_1q_m3(qc, h, simulator, shots=128, m3_cache=m3_cache)
        print("mthree: calibration complete (single-qubit)")
except Exception as e:
    print(f"M3 calibration warning: {e}")

sys.stdout.flush()

mitigation_schemes = {
    'raw': {'use_m3': False, 'use_dd': False},
    'm3_only': {'use_m3': True, 'use_dd': False},
    'm3_dd': {'use_m3': True, 'use_dd': True}
}

all_results = {}


def vqe_wrapper(kx, ky):
    return band_structure_vqe(
        kx, ky, t1, t2, M, a_vecs, b_vecs,
        gpu_id=None,
    )


for scheme_name, scheme_config in mitigation_schemes.items():
    print(f"\n{'=' * 20} RUNNING: {scheme_name.upper()} {'=' * 20}")
    start_time = time.time()

    from utils_test import set_mitigation_config

    set_mitigation_config(**scheme_config)

    n_workers = min(16, cpu_count)
    results = Parallel(n_jobs=n_workers, verbose=5, backend="threading")(
        delayed(vqe_wrapper)(kx, ky) for (kx, ky) in k_points
    )

    elapsed_time = time.time() - start_time
    print(f"\n{scheme_name.upper()} calculations completed in {elapsed_time:.2f} seconds")
    print(f"Average time per k-point: {elapsed_time / len(k_points):.2f} seconds")

    all_results[scheme_name] = {k: v for k, v in zip(k_points, results)}

print("\n" + "=" * 60)
print("ALL VQE CALCULATIONS COMPLETED")
print("=" * 60)

start_time = time.time()

results_exact = Parallel(n_jobs=-1, verbose=5)(
    delayed(band_structure_exact)(kx, ky, t1, t2, M, a_vecs, b_vecs)
    for (kx, ky) in k_points
)

elapsed_time = time.time() - start_time
print(f"Exact calculations completed in {elapsed_time:.2f} seconds")

exact_results = {k: v for k, v in zip(k_points, results_exact)}

comprehensive_data = {
    'metadata': {
        'n_sites': n_sites,
        't1': t1, 't2': t2, 'M': M,
        'samples': samples,
        'k_points': [list(k) for k in k_points],
        'mitigation_schemes': list(mitigation_schemes.keys())
    },
    'exact': {str(k): v for k, v in exact_results.items()},
    **{scheme: {str(k): v for k, v in results.items()}
       for scheme, results in all_results.items()}
}

file_path = os.path.join(
    os.getcwd(), "..", "..",
    f"cache/haldane-model/band-structure/{n_sites}-sites/comprehensive-noisy-{samples}-samples.json",
)
os.makedirs(os.path.dirname(file_path), exist_ok=True)
with open(file_path, "w") as f:
    json.dump(comprehensive_data, f, indent=4)

x_list = np.linspace(-np.pi, np.pi, samples)
y_list = np.linspace(-np.pi, np.pi, samples)
kx_vals, ky_vals = np.meshgrid(x_list, y_list, indexing='xy')

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

schemes_to_plot = ['raw', 'm3_only', 'm3_dd']
scheme_labels = ['Raw (No Mitigation)', 'M3 Readout Correction', 'M3 + Dynamical Decoupling']

for i, (scheme, label) in enumerate(zip(schemes_to_plot, scheme_labels)):
    error_grid = error_stats[scheme]['errors'].reshape((samples, samples))

    im = axes[i].imshow(error_grid, origin='lower', cmap='Reds',
                        extent=[kx_vals.min(), kx_vals.max(), ky_vals.min(), ky_vals.max()],
                        aspect='auto')
    plt.colorbar(im, ax=axes[i])
    axes[i].set_title(f"{label}\nMean Error: {error_stats[scheme]['mean']:.4f}")
    axes[i].set_xlabel("$k_x$")
    axes[i].set_ylabel("$k_y$")

improvement_grid = ((error_stats['raw']['errors'] - error_stats['m3_dd']['errors']) /
                    error_stats['raw']['errors'] * 100).reshape((samples, samples))
im = axes[3].imshow(improvement_grid, origin='lower', cmap='RdYlGn',
                    extent=[kx_vals.min(), kx_vals.max(), ky_vals.min(), ky_vals.max()],
                    aspect='auto')
plt.colorbar(im, ax=axes[3], label='% Improvement')
axes[3].set_title(f"Error Reduction: M3+DD vs Raw\nMean: {np.mean(improvement_grid):.1f}%")
axes[3].set_xlabel("$k_x$")
axes[3].set_ylabel("$k_y$")

plt.suptitle(f"Error Mitigation Comparison - Haldane Model ({samples}² samples)\n"
             f"$t_1={t1}, t_2={t2}, M={M}, N_{{sites}}={n_sites}$", fontsize=14)
plt.tight_layout()

error_comparison_path = os.path.join(
    os.getcwd(), "..", "..",
    f"plots/haldane-model/band-structure/{n_sites}-sites/error-mitigation-comparison-{samples}-samples.png"
)
os.makedirs(os.path.dirname(error_comparison_path), exist_ok=True)
plt.savefig(error_comparison_path, dpi=150, bbox_inches='tight')
print(f"Error comparison plot saved to: {error_comparison_path}")

print("Creating 3D band structure plot (best mitigation)...")
best_data = all_results['m3_dd']
E_plus = np.array([-best_data[key] for key in best_data]).reshape((samples, samples))
E_minus = np.array([best_data[key] for key in best_data]).reshape((samples, samples))

diverging_cmap = LinearSegmentedColormap.from_list('blue_white_red', ['royalblue', 'white', 'red'])
norm = TwoSlopeNorm(vmin=np.min([E_minus, E_plus]), vcenter=0, vmax=np.max([E_minus, E_plus]))

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(kx_vals, ky_vals, E_plus, facecolors=diverging_cmap(norm(E_plus)),
                edgecolor='k', linewidth=0.2, alpha=0.8, antialiased=True)
ax.plot_surface(kx_vals, ky_vals, E_minus, facecolors=diverging_cmap(norm(E_minus)),
                edgecolor='k', linewidth=0.2, alpha=0.8, antialiased=True)

x_ticks = np.arange(-np.pi, np.pi + 0.1, np.pi)
y_ticks = np.arange(-np.pi, np.pi + 0.1, np.pi)
ax.set_xticks(x_ticks)
ax.set_yticks(y_ticks)
ax.set_xticklabels([r'$-\pi$', r'$0$', r'$\pi$'])
ax.set_yticklabels([r'$-\pi$', r'$0$', r'$\pi$'])

ax.set_xlabel('$k_x$')
ax.set_ylabel('$k_y$')
ax.set_zlabel('$E(k)$')

ax.set_title(f"Haldane Model Band Structure (VQE + M3 + DD, {samples}² samples)\n"
             f"$t_1={t1}, t_2={t2}, M={M}, N_{{sites}}={n_sites}$")
ax.view_init(elev=20, azim=45)

band_3d_path = os.path.join(
    os.getcwd(), "..", "..",
    f"plots/haldane-model/band-structure/{n_sites}-sites/mitigated-band-structure-3d-{samples}-samples.png"
)
plt.savefig(band_3d_path, dpi=150, bbox_inches='tight')
print(f"3D band structure plot saved to: {band_3d_path}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

schemes = ['Raw', 'M3 Only', 'M3 + DD']
means = [error_stats['raw']['mean'], error_stats['m3_only']['mean'], error_stats['m3_dd']['mean']]
stds = [error_stats['raw']['std'], error_stats['m3_only']['std'], error_stats['m3_dd']['std']]

colors = ['red', 'orange', 'green']
bars1 = ax1.bar(schemes, means, yerr=stds, capsize=5, color=colors, alpha=0.7, edgecolor='black')
ax1.set_ylabel('Mean Absolute Error')
ax1.set_title('Mean Error by Mitigation Scheme')
ax1.grid(True, alpha=0.3)

for bar, mean in zip(bars1, means):
    height = bar.get_height()
ax1.text(bar.get_x() + bar.get_width() / 2., height + max(stds) * 0.05,
         f'{mean:.4f}', ha='center', va='bottom', fontweight='bold')

error_data = [error_stats['raw']['errors'], error_stats['m3_only']['errors'], error_stats['m3_dd']['errors']]
bp = ax2.boxplot(error_data, labels=schemes, patch_artist=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
patch.set_alpha(0.7)

ax2.set_ylabel('Absolute Error')
ax2.set_title('Error Distribution by Mitigation Scheme')
ax2.grid(True, alpha=0.3)

plt.suptitle(f"Error Mitigation Performance Statistics ({samples}² k-points)", fontsize=14)
plt.tight_layout()

stats_path = os.path.join(
    os.getcwd(), "..", "..",
    f"plots/haldane-model/band-structure/{n_sites}-sites/error-statistics-{samples}-samples.png"
)
plt.savefig(stats_path, dpi=150, bbox_inches='tight')
print(f"Error statistics plot saved to: {stats_path}")

print("\n" + "=" * 60)
print("SIMULATION COMPLETED SUCCESSFULLY")
print("=" * 60)
print(f"\nKey Results:")
print(f"  - Raw VQE mean error: {raw_mean:.4f}")
print(f"  - M3-corrected mean error: {m3_mean:.4f} ({((raw_mean - m3_mean) / raw_mean * 100):+.1f}%)")
print(f"  - M3+DD mean error: {m3dd_mean:.4f} ({((raw_mean - m3dd_mean) / raw_mean * 100):+.1f}%)")
print(
    f"  - Best improvement: {max((raw_mean - m3_mean) / raw_mean * 100, (raw_mean - m3dd_mean) / raw_mean * 100):.1f}%")
print("=" * 60)