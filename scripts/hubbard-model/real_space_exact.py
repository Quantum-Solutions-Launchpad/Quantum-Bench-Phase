from utils import hubbard_real_space_exact, setup_logging
import numpy as np
import matplotlib.pyplot as plt
import os

setup_logging()

n_sites = 6
t, U = 1.0, 1.0
spin = 2

data = {}
for n_occ in range(spin*n_sites+1):
    data[n_occ] = hubbard_real_space_exact(n_sites, t, U, n_occ)

plt.figure()
plt.plot(range(spin*n_sites+1), data.values(), 'ro-')
plt.xlabel("Particle Number")
plt.ylabel("$E$")
plt.title("Real Space Hubbard Hamiltonian Ground State Energy (Exact)\n$t="+str(t)+", U="+str(U)+", N_{\\text{sites}}="+str(n_sites)+"$")
plt.grid(True, alpha=0.3)
plt.tight_layout()

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
file_path = os.path.join(project_root, "plots/hubbard-model/"+str(n_sites)+"-sites/exact-U-"+str(U)+".png")
os.makedirs(os.path.dirname(file_path), exist_ok=True)
plt.savefig(file_path)
