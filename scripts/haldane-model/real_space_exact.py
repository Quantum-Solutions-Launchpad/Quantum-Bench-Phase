from utils import real_space_exact, setup_logging
import numpy as np
import matplotlib.pyplot as plt
import os

setup_logging()

n_sites = 4
t1, t2, phi = 1.0, 0.05, np.pi/4
spin = 2

data = {}
for n_occ in range(spin*n_sites+1):
    data[n_occ] = real_space_exact(n_sites, t1, t2, phi, n_occ)

plt.figure()
plt.plot(range(spin*n_sites+1), data.values(), 'ro-')
plt.xlabel("Particle Number")
plt.ylabel("$E$")
plt.title("Real Space Haldane Hamiltonian Ground State Energy (Exact)\n$t_1="+str(t1)+", t_2="+str(t2)+", \\phi=\\pi/"+str(int(np.pi/phi))+", N_{\\text{sites}}="+str(n_sites)+"$")
plt.tight_layout()

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
file_path = os.path.join(project_root, "plots/haldane-model/real-space/"+str(n_sites)+"-sites/exact.png")
plt.savefig(file_path)