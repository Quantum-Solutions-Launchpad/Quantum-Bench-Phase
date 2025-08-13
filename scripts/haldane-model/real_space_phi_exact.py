from utils import real_space_exact
import numpy as np
import matplotlib.pyplot as plt
import os

n_sites, n_occ = 6, 9
t1, t2 = 1.0, 0.05
spin = 2

data = {}
for phi in np.linspace(0, 2*np.pi, 12):
    data[phi] = real_space_exact(n_sites, t1, t2, phi, n_occ)

plt.figure()
plt.plot(np.linspace(0, 2*np.pi, 12), data.values(), 'ro-')
plt.xlabel("$\\phi$")
plt.ylabel("$E$")
plt.title("Real Space Haldane Hamiltonian Ground State Energy (Exact)\n$t_1="+str(t1)+", t_2="+str(t2)+", N_{\\text{sites}}="+str(n_sites)+", N_{\\text{occ}}="+str(n_occ)+"$")
plt.tight_layout()

file_path = os.path.join(os.getcwd(), "..", "..", "plots/haldane-model/real-space/exact-phi.png")
plt.savefig(file_path)