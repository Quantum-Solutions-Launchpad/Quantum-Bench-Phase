# Haldane Model Band Structure Simulation
import numpy as np
import matplotlib.pyplot as plt

##constants##
t = 1.0              ##nearest-neighbor hopping##
t2 = 0.1            ##next-nearest-neighbor hopping##
phi = np.pi/2       ##complex phase##
M = 0.0             ##sublattice mass term##

##lattice vectors##
a1 = np.array([1.0, 0.0])
a2 = np.array([0.5, np.sqrt(3)/2])

##reciprocal lattice vectors##
b1 = 2 * np.pi * np.array([1, -1/np.sqrt(3)])
b2 = 2 * np.pi * np.array([0, 2/np.sqrt(3)])

##momentum space grid##
kx_vals = np.linspace(-np.pi, np.pi, 300)
ky_vals = np.linspace(-np.pi, np.pi, 300)

##nearest neighbor vectors##
d1 = np.array([0.0, -1.0])
d2 = np.array([np.sqrt(3)/2, 0.5])
d3 = np.array([-np.sqrt(3)/2, 0.5])

##next-nearest neighbor vectors##
nn1 = d2 - d3
nn2 = d3 - d1
nn3 = d1 - d2

##band energies##
E_plus = np.zeros((len(kx_vals), len(ky_vals)))
E_minus = np.zeros_like(E_plus)

for i, kx in enumerate(kx_vals):
    for j, ky in enumerate(ky_vals):
        k = np.array([kx, ky])
        f = 0
        for d in [d1, d2, d3]:
            f += np.exp(1j * np.dot(k, d))
        
        d_x = t * np.real(f)
        d_y = -t * np.imag(f)

        d_z = M
        for d in [nn1, nn2, nn3]:
            d_z += 2 * t2 * np.sin(phi) * np.sin(np.dot(k, d))

        E = np.sqrt(d_x**2 + d_y**2 + d_z**2)
        E_plus[i, j] = E
        E_minus[i, j] = -E

##plotting##
fig, ax = plt.subplots(figsize=(8,6))
X, Y = np.meshgrid(kx_vals, ky_vals)
c = ax.contourf(X, Y, E_plus.T, levels=50, cmap='viridis')
plt.colorbar(c, ax=ax)
ax.set_title("Haldane Model Upper Band")
ax.set_xlabel("$k_x$")
ax.set_ylabel("$k_y$")
plt.tight_layout()
plt.show()
