import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from band_structure_utils import band_structure_vqe as haldane_band_structure_vqe, band_structure_exact as haldane_band_structure_exact
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
import json
from itertools import product
from joblib import Parallel, delayed
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
import argparse

n_sites = 4
t1, t2, M = 1.0, 0.05, 0.2
a_vecs = {
    4: [np.array([0.0, -1.0]), np.array([0.0, 1.0]), np.array([1.0, 0.0]), np.array([-1.0, 0.0])],
    6: [np.array([0.0, -1.0]), np.array([np.sqrt(3)/2, 0.5]), np.array([-np.sqrt(3)/2, 0.5])]
}.get(n_sites, [])
b_vecs = {
    4: [np.array([-1.0, -1.0]), np.array([1.0, -1.0]), np.array([-1.0, 1.0]), np.array([1.0, 1.0])],
    6: [a_vecs[1]-a_vecs[2], a_vecs[2]-a_vecs[0], a_vecs[0]-a_vecs[1]]
}.get(n_sites, [])
samples = 25
backend = FakeSherbrooke()

parser = argparse.ArgumentParser()
parser.add_argument("--no-debug", action="store_true", help="Suppress debug logs")
args = parser.parse_args()

x_list = [float(kx) for kx in np.linspace(-np.pi, np.pi, samples)]
y_list = [float(ky) for ky in np.linspace(-np.pi, np.pi, samples)]
k_points = list(product(x_list, y_list))

def tagged_vqe(k_point, *vqe_args, **vqe_kwargs):
    return k_point, haldane_band_structure_vqe(*vqe_args, **vqe_kwargs)

def init_worker_logging():
    from core import setup_logging
    setup_logging(debug_enabled=not args.no_debug)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
raw_data_path = os.path.join(project_root, f"logs/haldane/band-structure/{n_sites}-sites/raw-data/simulated-noisy-{samples}-samples.json")
os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)

raw_data = {
    "parameters": {
        "n_sites": n_sites,
        "t1": t1, "t2": t2, "M": M,
        "samples": samples,
        "simulation": "noisy"
    },
    "k_point_energies": {}
}

with open(raw_data_path, "w") as f:
    json.dump(raw_data, f, indent=4)

for k_point, energy in Parallel(n_jobs=-1, return_as="generator_unordered", initializer=init_worker_logging)(
    delayed(tagged_vqe)(kpt, kpt[0], kpt[1], t1, t2, M, a_vecs, b_vecs, backend)
    for kpt in k_points
):
    raw_data["k_point_energies"][str(k_point)] = energy
    with open(raw_data_path, "w") as f:
        json.dump(raw_data, f, indent=4)

data = {kpt: raw_data["k_point_energies"][str(kpt)] for kpt in k_points}

stringified_data = {str(k): v for k, v in data.items()}
final_path = os.path.join(project_root, f"logs/haldane/band-structure/{n_sites}-sites/simulated-noisy-{samples}-samples.json")
os.makedirs(os.path.dirname(final_path), exist_ok=True)
with open(final_path, "w") as f:
    json.dump(stringified_data, f, indent=4)

x_list = np.linspace(-np.pi, np.pi, samples)
y_list = np.linspace(-np.pi, np.pi, samples)
kx_vals, ky_vals = np.meshgrid(x_list, y_list, indexing='xy')
E_plus = np.array([-data[key] for key in data]).reshape((samples, samples))
E_minus = np.array([data[key] for key in data]).reshape((samples, samples))

diverging_cmap = LinearSegmentedColormap.from_list(
    'blue_white_red',
    ['royalblue', 'white', 'red']
)
norm = TwoSlopeNorm(vmin=np.min([E_minus, E_plus]), vcenter=0, vmax=np.max([E_minus, E_plus]))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(kx_vals, ky_vals, E_plus, facecolors=diverging_cmap(norm(E_plus)), edgecolor='k', linewidth=0.2, antialiased=True)
ax.plot_surface(kx_vals, ky_vals, E_minus, facecolors=diverging_cmap(norm(E_minus)), edgecolor='k', linewidth=0.2, antialiased=True)

x_ticks = np.arange(np.floor(x_list.min()/np.pi)*np.pi, np.ceil(x_list.max()/np.pi)*np.pi+0.1, np.pi)
y_ticks = np.arange(np.floor(y_list.min()/np.pi)*np.pi, np.ceil(y_list.max()/np.pi)*np.pi+0.1, np.pi)
ax.set_xticks(x_ticks)
ax.set_yticks(y_ticks)
ax.set_xticklabels([rf'${int(t/np.pi)}\pi$' if t not in (0, np.pi, -np.pi) else (r'$0$' if t==0 else (r'$-\pi$' if t<0 else r'$\pi$')) for t in x_ticks])
ax.set_yticklabels([rf'${int(t/np.pi)}\pi$' if t not in (0, np.pi, -np.pi) else (r'$0$' if t==0 else (r'$-\pi$' if t<0 else r'$\pi$')) for t in y_ticks])

ax.set_xlabel('$k_x$')
ax.set_ylabel('$k_y$')
ax.set_zlabel('$E(k)$')
ax.set_title("Haldane Model Band Structure (VQE, Qiskit Aer Noisy, $"+str(samples)+"^2$ samples)\n$t_1="+str(t1)+", t_2="+str(t2)+", M="+str(M)+", N_{\\text{sites}}="+str(n_sites)+"$")
ax.view_init(elev=20)

file_path = os.path.join(project_root, "plots/haldane/band-structure/"+str(n_sites)+"-sites/simulated-noisy-"+str(samples)+"-samples-3d.png")
plt.savefig(file_path)

fig, ax = plt.subplots(1, 2, figsize=(14,6))

c1 = ax[0].imshow(E_plus, origin='lower', cmap='viridis', extent=[kx_vals.min(), kx_vals.max(), ky_vals.min(), ky_vals.max()], aspect='auto')
plt.colorbar(c1, ax=ax[0])
ax[0].set_title("Upper Band: $E_+(k)$")
ax[0].set_xlabel("$k_x$")
ax[0].set_ylabel("$k_y$")

c2 = ax[1].imshow(E_minus, origin='lower', cmap='plasma', extent=[kx_vals.min(), kx_vals.max(), ky_vals.min(), ky_vals.max()], aspect='auto')
plt.colorbar(c2, ax=ax[1])
ax[1].set_title("Lower Band: $E_-(k)$")
ax[1].set_xlabel("$k_x$")
ax[1].set_ylabel("$k_y$")

fig.suptitle("Haldane Model Band Structure (VQE, Qiskit Aer Noisy, $"+str(samples)+"^2$ samples)\n$t_1="+str(t1)+", t_2="+str(t2)+", M="+str(M)+", N_{\\text{sites}}="+str(n_sites)+"$", fontsize=16)

plt.tight_layout()
file_path = os.path.join(project_root, "plots/haldane/band-structure/"+str(n_sites)+"-sites/simulated-noisy-"+str(samples)+"-samples-heatmap.png")
plt.savefig(file_path)

results = Parallel(n_jobs=-1)(
    delayed(haldane_band_structure_exact)(kx, ky, t1, t2, M, a_vecs, b_vecs)
    for kx, ky in k_points
)
analytic = {k: v for k, v in zip(k_points, results)}
error_data = {k: abs(data[k]-analytic[k]) for k in k_points}
error = np.array([error_data[key] for key in error_data]).reshape((samples, samples))

fig, ax = plt.subplots()

c = ax.imshow(error, origin='lower', cmap='Reds', extent=[kx_vals.min(), kx_vals.max(), ky_vals.min(), ky_vals.max()], aspect='auto')
plt.colorbar(c)
ax.set_xlabel("$k_x$")
ax.set_ylabel("$k_y$")

fig.suptitle("Haldane Model Band Structure Absolute Error (VQE, Qiskit Aer Noisy, $"+str(samples)+"^2$ samples)\n$t_1="+str(t1)+", t_2="+str(t2)+", M="+str(M)+", N_{\\text{sites}}="+str(n_sites)+"$", fontsize=11)

file_path = os.path.join(project_root, "plots/haldane/band-structure/"+str(n_sites)+"-sites/simulated-noisy-"+str(samples)+"-samples-error.png")
plt.savefig(file_path)
