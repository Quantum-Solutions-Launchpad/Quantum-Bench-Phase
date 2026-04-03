import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D


##parameters##
Lx = 30
Ly = 30
J = {'x': 1.0, 'y': 1.0, 'z': 4.0}
N = Lx * Ly

##build honeycomb lattice##
def honeycomb_bonds(Lx, Ly):
    bonds = []
    for x in range(Lx):
        for y in range(Ly):
            i = 2 * (x + Lx * y)
            j = i + 1
            bonds.append((i, j, 'z'))
            if x + 1 < Lx:
                bonds.append((j, 2 * ((x + 1) + Lx * y), 'x'))
            if y + 1 < Ly:
                bonds.append((j, 2 * (x + Lx * (y + 1)), 'y'))
    return bonds

def build_majorana_hamiltonian(n_sites, bonds, vortex_config=None):
    H = np.zeros((n_sites, n_sites), dtype=np.complex128)
    for (i, j, bond_type) in bonds:
        sign = -1
        if vortex_config and ((i, j) in vortex_config or (j, i) in vortex_config):
            sign *= -1
        print(f"bond_type={bond_type}, J[bond_type]={J[bond_type]}, type={type(J[bond_type])}")
        value = 1j * sign * J[bond_type]
        H[i, j] = value
        H[j, i] = -value
    return H



##convert Majorana H to BdG form##
def majorana_to_bdg(H_majorana):
    N_majorana = H_majorana.shape[0]
    assert N_majorana % 2 == 0
    n = N_majorana // 2  ## Number of complex fermions

    h = np.zeros((n, n), dtype=complex)
    delta = np.zeros((n, n), dtype=complex)

    for i in range(n):
        for j in range(n):
            A = H_majorana[2*i, 2*j]
            B = H_majorana[2*i, 2*j+1]
            C = H_majorana[2*i+1, 2*j]
            D = H_majorana[2*i+1, 2*j+1]

            h[i, j] = 0.5 * (D - A + 1j * (B + C))
            delta[i, j] = 0.5 * (D + A - 1j * (B - C))

    ##BdG matrix##
    top = np.hstack((h, delta))
    bottom = np.hstack((-delta.conj(), -h.T.conj()))
    H_bdg = np.vstack((top, bottom))

    return H_bdg

def honeycomb_positions(Lx, Ly):
    positions = []
    a = 1  ## lattice spacing
    for x in range(Lx):
        for y in range(Ly):
            base_x = a * 3/2 * x
            base_y = a * np.sqrt(3) * y
            # Site 0 of unit cell
            positions.append((base_x, base_y))
            # Site 1 of unit cell
            positions.append((base_x + a/2, base_y + a * np.sqrt(3)/2))
    return np.array(positions)

##run test##
bonds = honeycomb_bonds(Lx, Ly)
n_sites = 2 * Lx * Ly
vortex_links = {(2,3), (2 + 2*4, 3 + 2*4)}  ##arbitrary vortex config for testing

H_maj = build_majorana_hamiltonian(n_sites, bonds, vortex_config=vortex_links)
H_bdg = majorana_to_bdg(H_maj)

##compare eigenvalues##
eig_maj = np.linalg.eigvalsh(H_maj)
eig_bdg = np.linalg.eigvalsh(H_bdg)

##sort and clean up small imaginary parts##
eig_maj = np.sort(np.real_if_close(eig_maj))
eig_bdg = np.sort(np.real_if_close(eig_bdg))

##plot##
plt.figure(figsize=(8, 5))
plt.plot(eig_maj, 'ko', label='Majorana Spectrum')
plt.plot(eig_bdg, 'r+', label='BdG Spectrum', markersize=8)
plt.title("Comparison of Majorana vs BdG Spectra")
plt.xlabel("Mode Index")
plt.ylabel("Energy")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 4))
plt.hist(eig_maj, bins=50, color='black', alpha=0.7)
plt.xlabel("Energy")
plt.ylabel("Number of Modes")
plt.title("Density of States (Majorana Spectrum)")
plt.grid(True)
plt.tight_layout()
plt.show()


## Extract zero mode ##
eigvals, eigvecs = np.linalg.eigh(H_maj)
zero_mode = eigvecs[:, len(eigvals)//2]  ## ε₀ mode
amplitudes = np.abs(zero_mode)

## Get 2D positions ##
positions = honeycomb_positions(Lx, Ly)
x = positions[:, 0]
y = positions[:, 1]

## Plot ##
plt.figure(figsize=(8, 6))
sc = plt.scatter(x, y, c=amplitudes, cmap='viridis', s=100, edgecolor='k')
plt.colorbar(sc, label='|ψ|')
plt.title("Majorana Zero Mode Amplitude on Honeycomb Lattice")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.grid(True)
plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

## Coordinates and amplitudes ##
x = positions[:, 0]
y = positions[:, 1]
z = np.abs(zero_mode)

## Plot as 3D scatter ##
sc = ax.scatter(x, y, z, c=z, cmap='cividis', s=50 + 300*z, edgecolor='k')

## Labels and View Settings ##
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("|ψ|")
ax.set_title("3D Majorana Zero Mode Density")
fig.colorbar(sc, ax=ax, shrink=0.6, label='|ψ| Amplitude')

ax.view_init(elev=30, azim=45)  # Adjust for better 3D perspective
plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

## Bar heights and bases ##
z_base = np.zeros_like(z)
dx = dy = 0.5 * np.ones_like(z)
dz = z

## 3D bar plot ##
ax.bar3d(x, y, z_base, dx, dy, dz, color=plt.cm.cividis(z / np.max(z)), edgecolor='k', alpha=0.9)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("|ψ|")
ax.set_title("3D Bar Plot of Majorana Zero Mode Density")
plt.tight_layout()
plt.show()

from matplotlib import cm

## Extract amplitudes and positions ##
zero_mode = eigvecs[:, len(eigvals)//2]
amplitudes = np.abs(zero_mode)
positions = honeycomb_positions(Lx, Ly)
x = positions[:, 0]
y = positions[:, 1]
z = amplitudes

## Create a grid over the 2D space ##
xi = np.linspace(x.min() - 0.5, x.max() + 0.5, 200)
yi = np.linspace(y.min() - 0.5, y.max() + 0.5, 200)
X, Y = np.meshgrid(xi, yi)

## Interpolate z values (wavefunction amplitudes) onto the grid ##
Z = griddata((x, y), z, (X, Y), method='cubic', fill_value=0.0)

## Plot the surface ##
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(X, Y, Z, cmap='coolwarm', edgecolor='none', antialiased=True)

## Labels and view settings ##
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('|ψ|')
ax.set_title('Smooth 3D Surface of Majorana Zero Mode Amplitude')
fig.colorbar(surf, ax=ax, shrink=0.6, label='|ψ| Amplitude')
ax.view_init(elev=30, azim=45)
plt.tight_layout()
plt.show()