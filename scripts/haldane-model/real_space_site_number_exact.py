from utils import real_space_exact
import numpy as np
import matplotlib.pyplot as plt
import os

t1, t2, phi = 1.0, 0.05, np.pi/4
spin = 2
n_sites_max = 10

data = {}
for n_sites in range(3, n_sites_max+1):
    data[n_sites] = real_space_exact(n_sites, t1, t2, phi, n_sites)

plt.figure()
plt.plot(range(3, n_sites_max+1), data.values(), 'ro-')
plt.xlabel("Number of Sites")
plt.ylabel("$E$")
plt.title("Real Space Haldane Hamiltonian Ground State Energy (Exact)")
plt.tight_layout()

file_path = os.path.join(os.getcwd(), "..", "..", "plots/haldane-model/real-space/exact-site-number.png")
plt.savefig(file_path)