from utils import real_space_exact, real_space_vqe, real_space_iqpe
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from qiskit_nature.second_q.mappers import JordanWignerMapper
from joblib import Parallel, delayed

n_sites = 6
t1, t2, phi = 1.0, 0.05, np.pi/4
spin = 2
mapper = JordanWignerMapper()
max_iters = 100
t, n_trot, n_iters = 0.2, 5, 8

jobs = []
for n_occ in range(spin * n_sites + 1):
    jobs.append(delayed(real_space_exact)(n_sites, t1, t2, phi, n_occ))
    jobs.append(delayed(real_space_vqe)(n_sites, t1, t2, phi, n_occ, mapper, max_iters))
    jobs.append(delayed(real_space_iqpe)(n_sites, t1, t2, phi, n_occ, mapper, t, n_trot, n_iters))

results = Parallel(n_jobs=-1)(jobs)
exact, vqe, iqpe = results[0::3], results[1::3], results[2::3]

data = {
    "exact": {i: exact[i] for i in range(spin*n_sites+1)},
    "vqe": {i: vqe[i] for i in range(spin*n_sites+1)},
    "iqpe": {i: iqpe[i] for i in range(spin*n_sites+1)},
    "vqe_error": {i: abs(vqe[i]-exact[i]) for i in range(spin*n_sites+1)},
    "iqpe_error": {i: abs(iqpe[i]-exact[i]) for i in range(spin*n_sites+1)}
}

file_path = os.path.join(os.getcwd(), "..", "..", "cache/haldane-model/real-space/"+str(n_sites)+"-sites/simulated-ideal-slater.json")
with open(file_path, "w") as f:
    json.dump(data, f, indent=4)

plt.figure()
plt.plot(range(spin*n_sites+1), data["exact"].values(), 'ro-', label="Exact")
plt.plot(range(spin*n_sites+1), data["vqe"].values(), 'bo', label=f"VQE (max_iters={max_iters})")
plt.plot(range(spin*n_sites+1), data["iqpe"].values(), 'go', label=f"IQPE (t={t}, n_trot={n_trot}, n_iters={n_iters})")
plt.legend()
plt.xlabel("Particle Number")
plt.ylabel("$E$")
plt.title("Real Space Haldane Hamiltonian Ground State Energy (Qiskit Aer Ideal)\n$t_1="+str(t1)+", t_2="+str(t2)+", \\phi=\\pi/"+str(int(np.pi/phi))+", N_{\\text{sites}}="+str(n_sites)+"$", fontsize=11)
plt.tight_layout()

file_path = os.path.join(os.getcwd(), "..", "..", "plots/haldane-model/real-space/"+str(n_sites)+"-sites/simulated-ideal-slater.png")
plt.savefig(file_path)

x = np.arange(spin*n_sites+1)
fig, ax = plt.subplots(layout='constrained')

ax.plot(x, data["vqe_error"].values(), 'o-', label="VQE", color="firebrick")
ax.plot(x, data["iqpe_error"].values(), 'o-', label="IQPE", color="lightcoral")

ax.set_xlabel("Particle Number")
ax.set_ylabel("Absolute Error")
ax.set_title("Real Space Haldane Hamiltonian Ground State Energy (Qiskit Aer Ideal)\n$t_1="+str(t1)+", t_2="+str(t2)+", \\phi=\\pi/"+str(int(np.pi/phi))+", N_{\\text{sites}}="+str(n_sites)+"$", fontsize=11)
ax.legend()

file_path = os.path.join(os.getcwd(), "..", "..", "plots/haldane-model/real-space/"+str(n_sites)+"-sites/simulated-ideal-slater-error.png")
plt.savefig(file_path)