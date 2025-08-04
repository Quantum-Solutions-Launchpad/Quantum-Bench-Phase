from utils import band_structure_vqe
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
import os

t1, t2, M = 1.0, 0.05, 0.2
a_vecs = [np.array([0.0, -1.0]), np.array([np.sqrt(3)/2, 0.5]), np.array([-np.sqrt(3)/2, 0.5])]
b_vecs = [a_vecs[1]-a_vecs[2], a_vecs[2]-a_vecs[0], a_vecs[0]-a_vecs[1]]
samples = 100

data = band_structure_vqe(t1, t2, M, a_vecs, b_vecs, samples)

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

ax.set_xticks([-np.pi, 0, np.pi])
ax.set_yticks([-np.pi, 0, np.pi])
ax.set_xticklabels([r'$-\pi$', r'$0$', r'$\pi$'])
ax.set_yticklabels([r'$-\pi$', r'$0$', r'$\pi$'])

ax.set_xlabel('$k_x$')
ax.set_ylabel('$k_y$')
ax.set_zlabel('$E(k)$')
ax.set_title("Haldane Model Band Structure (VQE, Qiskit Aer Ideal, $"+str(samples)+"^2$ samples)\n$t_1="+str(t1)+", t_2="+str(t2)+", M="+str(M)+"$")
ax.view_init(elev=20)

file_path = os.path.join(os.getcwd(), "..", "..", "plots/haldane-model/band-structure/simulated-ideal-"+str(samples)+"-samples-3d.png")
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

fig.suptitle("Haldane Model Band Structure (VQE, Qiskit Aer Ideal, $"+str(samples)+"^2$ samples)\n$t_1="+str(t1)+", t_2="+str(t2)+", M="+str(M)+"$", fontsize=16)

plt.tight_layout()
file_path = os.path.join(os.getcwd(), "..", "..", "plots/haldane-model/band-structure/simulated-ideal-"+str(samples)+"-samples-heatmap.png")
plt.savefig(file_path)