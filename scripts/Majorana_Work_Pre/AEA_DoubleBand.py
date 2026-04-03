# Haldane Model Using Explicit Hamiltonian from Image
import numpy as np
import matplotlib.pyplot as plt

##constants##
t = 1.0              ##NN hopping (sets energy scale)##
t2 = 0.1             ##NNN hopping##
phi = np.pi/2        ##complex phase for t2##
M = 0.0              ##sublattice mass term##

##NN and NNN vectors##
d1 = np.array([0.0, -1.0])
d2 = np.array([np.sqrt(3)/2, 0.5])
d3 = np.array([-np.sqrt(3)/2, 0.5])

b1 = d2 - d3
b2 = d3 - d1
b3 = d1 - d2
nnn_vectors = [b1, b2, b3]

##function to compute Haldane band energies##
def haldane_energy(kx, ky, t=1.0, t2=0.1, M=0.0):
    k = np.array([kx, ky])

    ## NN term: d_x and d_y (H_0(k)) ##
    f = 0
    for d in [d1, d2, d3]:
        f += np.exp(1j * np.dot(k, d))
    d_x = t * np.real(f)
    d_y = -t * np.imag(f)

    ## NNN and Mass term: d_z ##
    d_z = M
    for b in nnn_vectors:
        d_z += 2 * t2 * np.sin(np.dot(k, b))

    E = np.sqrt(d_x**2 + d_y**2 + d_z**2)
    return E, -E

##momentum grid##
kx_vals = np.linspace(-np.pi, np.pi, 300)
ky_vals = np.linspace(-np.pi, np.pi, 300)

E_plus = np.zeros((len(kx_vals), len(ky_vals)))
E_minus = np.zeros_like(E_plus)

for i, kx in enumerate(kx_vals):
    for j, ky in enumerate(ky_vals):
        E_plus[i, j], E_minus[i, j] = haldane_energy(kx, ky, t, t2, M)

##plotting both bands##
fig, ax = plt.subplots(1, 2, figsize=(14,6))
X, Y = np.meshgrid(kx_vals, ky_vals)

c1 = ax[0].contourf(X, Y, E_plus.T, levels=50, cmap='viridis')
plt.colorbar(c1, ax=ax[0])
ax[0].set_title("Upper Band: $E_+(k)$")
ax[0].set_xlabel("$k_x$")
ax[0].set_ylabel("$k_y$")

c2 = ax[1].contourf(X, Y, E_minus.T, levels=50, cmap='plasma')
plt.colorbar(c2, ax=ax[1])
ax[1].set_title("Lower Band: $E_-(k)$")
ax[1].set_xlabel("$k_x$")
ax[1].set_ylabel("$k_y$")

plt.tight_layout()
plt.show()
