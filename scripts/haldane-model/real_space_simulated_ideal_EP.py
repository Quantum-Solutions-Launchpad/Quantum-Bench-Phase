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
n_iters, n_layers, n_reps = 5000, 5, 20

jobs = []
for n_occ in range(spin * n_sites + 1):
    jobs.append(delayed(real_space_exact)(n_sites, t1, t2, phi, n_occ))
    for _ in range(n_reps):
        jobs.append(delayed(real_space_vqe)(n_sites, t1, t2, phi, n_occ, mapper, n_iters, n_layers=n_layers))

results = Parallel(n_jobs=-1)(jobs)
exact, vqe = results[0::n_reps+1], [min(results[i:i+n_reps]) for i in range(1, len(results), n_reps+1)]

data = {
    "exact": {i: exact[i] for i in range(spin*n_sites+1)},
    "vqe": {i: vqe[i] for i in range(spin*n_sites+1)}
}

file_path = os.path.join(os.getcwd(), "..", "..", "cache/haldane-model/real-space/"+str(n_sites)+"-sites/simulated-ideal-EP-t2-"+str(t2)+".json")
with open(file_path, "w") as f:
    json.dump(data, f, indent=4)

plt.figure()
plt.plot(range(spin*n_sites+1), data["exact"].values(), 'ro-', label="Exact")
plt.plot(range(spin*n_sites+1), data["vqe"].values(), 'bo', label=f"VQE (n_iters={n_iters}, n_layers={n_layers}, n_reps={n_reps})")
plt.legend()
plt.xlabel("Particle Number")
plt.ylabel("$E$")
plt.title("Real Space Haldane Hamiltonian Ground State Energy (Qiskit Aer Ideal)\n$t_1="+str(t1)+", t_2="+str(t2)+", \\phi=\\pi/"+str(int(np.pi/phi))+", N_{\\text{sites}}="+str(n_sites)+"$", fontsize=11)
plt.tight_layout()

file_path = os.path.join(os.getcwd(), "..", "..", "plots/haldane-model/real-space/"+str(n_sites)+"-sites/simulated-ideal-EP-t2-"+str(t2)+".png")
plt.savefig(file_path)