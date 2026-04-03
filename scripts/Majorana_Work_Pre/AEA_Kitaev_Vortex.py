# Kitaev Honeycomb Model: Visualize Zero Modes at Vortices
import numpy as np
import matplotlib.pyplot as plt

##helper to build 2-site unit cell lattice##
def build_honeycomb_lattice(n_cells):
    positions = []
    bonds = []
    site_idx = lambda x, y, subl: 2 * (x + y * n_cells) + subl

    for x in range(n_cells):
        for y in range(n_cells):
            A = site_idx(x, y, 0)
            B = site_idx(x, y, 1)
            positions.append((x, y))   # A
            positions.append((x + 0.5, y + np.sqrt(3)/6))  # B

            ##x-bond: A-B in same cell##
            bonds.append((A, B, 'x'))

            ##y-bond: B to A in +y direction##
            if y < n_cells - 1:
                A_up = site_idx(x, y + 1, 0)
                bonds.append((B, A_up, 'y'))

            ##z-bond: B to A in +x direction##
            if x < n_cells - 1:
                A_right = site_idx(x + 1, y, 0)
                bonds.append((B, A_right, 'z'))

    return positions, bonds

##construct quadratic Majorana Hamiltonian##
def build_majorana_hamiltonian(n_sites, bonds, Jx=1.0, Jy=1.0, Jz=2.5, vortex_config=None):
    H = np.zeros((n_sites, n_sites), dtype=np.complex128)
    J = {'x': Jx, 'y': Jy, 'z': Jz}

    for (i, j, bond_type) in bonds:
        sign = 1.0
        if vortex_config and ((i, j) in vortex_config or (j, i) in vortex_config):
            sign = -1.0
        H[i, j] = 1j * sign * J[bond_type]
        H[j, i] = -1j * sign * J[bond_type]

    return H

##parameters##
n_cells = 5  # 5x5 = 50 sites
positions, bonds = build_honeycomb_lattice(n_cells)
n_sites = len(positions)

##insert two vortices by flipping two bond signs##
vortex_links = {
    (bonds[12][0], bonds[12][1]),  # pick bond in one plaquette
    (bonds[30][0], bonds[30][1])   # pick bond far from it
}

##build and diagonalize##
H = build_majorana_hamiltonian(n_sites, bonds, vortex_config=vortex_links)
evals, evecs = np.linalg.eigh(H)
evals = np.real(evals)

##identify zero mode (lowest magnitude eigenvalue)##
sorted_indices = np.argsort(np.abs(evals))
zero_mode_index = sorted_indices[0]
zero_mode = evecs[:, zero_mode_index]

##compute spatial density##
density = np.abs(zero_mode)**2
positions = np.array(positions)

##plot spatial density of zero mode##
plt.figure(figsize=(6,6))
plt.scatter(positions[:,0], positions[:,1], c=density, cmap='inferno', s=100, edgecolor='black')
plt.colorbar(label=r'$|\psi_i|^2$')
plt.title("Spatial Profile of Majorana Zero Mode (Two Vortices)")
plt.axis('equal')
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.tight_layout()
plt.show()