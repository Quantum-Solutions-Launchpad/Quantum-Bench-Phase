import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.haldane import band_structure_exact as haldane_band_structure_exact
from core import setup_logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
import os

setup_logging()

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
samples = 1000

x_list = [float(kx) for kx in np.linspace(-np.pi, np.pi, samples)]
y_list = [float(ky) for ky in np.linspace(-np.pi, np.pi, samples)]

data = {}
for kx in x_list:
    for ky in y_list:
        data[(kx, ky)] = haldane_band_structure_exact(kx, ky, t1, t2, M, a_vecs, b_vecs)

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

x_ticks = np.arange(np.floor(min(x_list)/np.pi)*np.pi, np.ceil(max(x_list)/np.pi)*np.pi+0.1, np.pi)
y_ticks = np.arange(np.floor(min(y_list)/np.pi)*np.pi, np.ceil(max(y_list)/np.pi)*np.pi+0.1, np.pi)
ax.set_xticks(x_ticks)
ax.set_yticks(y_ticks)
ax.set_xticklabels([rf'${int(t/np.pi)}\pi$' if t not in (0, np.pi, -np.pi) else (r'$0$' if t==0 else (r'$-\pi$' if t<0 else r'$\pi$')) for t in x_ticks])
ax.set_yticklabels([rf'${int(t/np.pi)}\pi$' if t not in (0, np.pi, -np.pi) else (r'$0$' if t==0 else (r'$-\pi$' if t<0 else r'$\pi$')) for t in y_ticks])

ax.set_xlabel('$k_x$')
ax.set_ylabel('$k_y$')
ax.set_zlabel('$E(k)$')
ax.set_title("Haldane Model Band Structure (Exact)\n$t_1="+str(t1)+", t_2="+str(t2)+", M="+str(M)+", N_{\\text{sites}}="+str(n_sites)+"$")
ax.view_init(elev=20)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
file_path = os.path.join(project_root, "plots/haldane/band-structure/"+str(n_sites)+"-sites/exact-3d.png")
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

fig.suptitle("Haldane Model Band Structure (Exact)\n$t_1="+str(t1)+", t_2="+str(t2)+", M="+str(M)+", N_{\\text{sites}}="+str(n_sites)+"$", fontsize=16)

plt.tight_layout()
file_path = os.path.join(project_root, "plots/haldane/band-structure/"+str(n_sites)+"-sites/exact-heatmap.png")
plt.savefig(file_path)