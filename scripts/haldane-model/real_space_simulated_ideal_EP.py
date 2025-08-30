from utils import real_space_exact, real_space_vqe
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from qiskit_nature.second_q.mappers import JordanWignerMapper
from joblib import Parallel, delayed

n_sites = 4
t1, t2, phi = 1.0, 0.0, np.pi/4
spin = 2
mapper = JordanWignerMapper()
max_iters = 20000

jobs = []
for n_occ in range(spin * n_sites + 1):
    jobs.append(delayed(real_space_exact)(n_sites, t1, t2, phi, n_occ))
    jobs.append(delayed(real_space_vqe)(n_sites, t1, t2, phi, n_occ, mapper, max_iters // 4, n_layers=5))
    jobs.append(delayed(real_space_vqe)(n_sites, t1, t2, phi, n_occ, mapper, max_iters // 2, n_layers=5))
    jobs.append(delayed(real_space_vqe)(n_sites, t1, t2, phi, n_occ, mapper, max_iters, n_layers=5))

results = Parallel(n_jobs=-1)(jobs)
exact, vqe1, vqe2, vqe3 = results[0::4], results[1::4], results[2::4], results[3::4]

data = {
    "exact": {i: exact[i] for i in range(spin*n_sites+1)},
    "vqe1": {i: vqe1[i] for i in range(spin*n_sites+1)},
    "vqe2": {i: vqe2[i] for i in range(spin*n_sites+1)},
    "vqe3": {i: vqe3[i] for i in range(spin*n_sites+1)}
}

file_path = os.path.join(os.getcwd(), "..", "..", "cache/haldane-model/real-space/"+str(n_sites)+"-sites/simulated-ideal-EP.json")
with open(file_path, "w") as f:
    json.dump(data, f, indent=4)

plt.figure()
plt.plot(range(spin*n_sites+1), data["exact"].values(), 'ro-', label="Exact")
plt.plot(range(spin*n_sites+1), data["vqe1"].values(), 'mo', label=f"VQE (max_iters={max_iters // 4})")
plt.plot(range(spin*n_sites+1), data["vqe2"].values(), 'go', label=f"VQE (max_iters={max_iters // 2})")
plt.plot(range(spin*n_sites+1), data["vqe3"].values(), 'bo', label=f"VQE (max_iters={max_iters})")
plt.legend()
plt.xlabel("Particle Number")
plt.ylabel("$E$")
plt.title("Real Space Haldane Hamiltonian Ground State Energy (Qiskit Aer Ideal)\n$t_1="+str(t1)+", t_2="+str(t2)+", \\phi=\\pi/"+str(int(np.pi/phi))+", N_{\\text{sites}}="+str(n_sites)+"$", fontsize=11)
plt.tight_layout()

file_path = os.path.join(os.getcwd(), "..", "..", "plots/haldane-model/real-space/"+str(n_sites)+"-sites/simulated-ideal-EP.png")
plt.savefig(file_path)