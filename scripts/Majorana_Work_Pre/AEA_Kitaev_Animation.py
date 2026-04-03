# Kitaev Honeycomb Model: Zero Mode Evolution with Lattice Size
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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
def build_majorana_hamiltonian(n_sites, bonds, Jx=1.0, Jy=1.0, Jz=1.0, vortex_config=None):
    H = np.zeros((n_sites, n_sites), dtype=np.complex128)
    J = {'x': Jx, 'y': Jy, 'z': Jz}

    for (i, j, bond_type) in bonds:
        sign = 1.0
        if vortex_config and (i, j) in vortex_config:
            sign = -1.0  # flip sign to insert a vortex
        H[i, j] = 1j * sign * J[bond_type]
        H[j, i] = -1j * sign * J[bond_type]  # antisymmetric

    return H

##animation setup##
fig, ax = plt.subplots(figsize=(6,4))
line, = ax.plot([], [], 'o', color='black')
ax.axhline(0, color='gray', linestyle='--', linewidth=0.7)
ax.set_title("Majorana Spectrum vs Lattice Size")
ax.set_xlabel("Eigenstate Index")
ax.set_ylabel("Energy")
ax.grid(True)

##update function##
def update(frame):
    ax.clear()
    n_cells = frame
    positions, bonds = build_honeycomb_lattice(n_cells)
    n_sites = len(positions)
    vortex_links = {(bonds[5][0], bonds[5][1])} if len(bonds) > 5 else set()
    H = build_majorana_hamiltonian(n_sites, bonds, vortex_config=vortex_links)
    evals, _ = np.linalg.eigh(H)
    evals = np.sort(np.real(evals))
    ax.plot(evals, 'o', color='black')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.7)
    ax.set_title(f"Lattice Size: {n_cells}x{n_cells} ({2*n_cells**2} sites)")
    ax.set_xlabel("Eigenstate Index")
    ax.set_ylabel("Energy")
    ax.set_ylim(-3, 3)
    ax.grid(True)

ani = animation.FuncAnimation(fig, update, frames=range(2, 7), interval=1000, repeat=True)
plt.tight_layout()
plt.show()
