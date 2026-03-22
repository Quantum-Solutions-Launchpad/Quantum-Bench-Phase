from utils import haldane_hubbard_real_space_exact, setup_logging
import numpy as np
import matplotlib.pyplot as plt
import os

setup_logging()

n_sites = 6
t1, U = 1.0, 1.0
t2, phi, M = 0.5, np.pi/4, 0.0
spin = 2

data = {}
for n_occ in range(spin*n_sites+1):
    data[n_occ] = haldane_hubbard_real_space_exact(n_sites, t1, U, t2, phi, M, n_occ)

plt.figure()
plt.plot(range(spin*n_sites+1), data.values(), 'ro-')
plt.xlabel("Particle Number")
plt.ylabel("$E$")
plt.title("Real Space Haldane–Hubbard Hamiltonian Ground State Energy (Exact)\n$t_1="+str(t1)+", U="+str(U)+", t_2="+str(t2)+", N_{\\text{sites}}="+str(n_sites)+"$")
plt.grid(True, alpha=0.3)
plt.tight_layout()

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
file_path = os.path.join(project_root, "plots/haldane-hubbard-model/"+str(n_sites)+"-sites/exact-U-"+str(U)+"-t2-"+str(t2)+".png")
os.makedirs(os.path.dirname(file_path), exist_ok=True)
plt.savefig(file_path)
