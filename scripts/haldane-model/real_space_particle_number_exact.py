from utils import real_space_exact
import numpy as np
import matplotlib.pyplot as plt
import os

n_sites = 6
t1, t2, phi = 1.0, 0.05, np.pi/4
spin = 2

data = {}
for n_occ in range(spin*n_sites+1):
    data[n_occ] = real_space_exact(n_sites, t1, t2, phi, n_occ)

plt.figure()
plt.plot(range(spin*n_sites+1), data.values(), 'ro-')
plt.xlabel("Particle Number")
plt.ylabel("$E$")
plt.title("Real Space Haldane Hamiltonian Ground State Energy (Exact)")
plt.tight_layout()

file_path = os.path.join(os.getcwd(), "..", "..", "plots/haldane-model/real-space/exact-particle-number-"+str(n_sites)+"-sites.png")
plt.savefig(file_path)