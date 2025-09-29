from utils import real_space_exact, real_space_iqpe, real_space_vqe
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from qiskit_nature.second_q.mappers import JordanWignerMapper
from joblib import Parallel, delayed

n_sites = 4
t1, t2, phi = 1.0, 0.05, np.pi/4
spin = 2
mapper = JordanWignerMapper()
vqe_iters, vqe_layers, vqe_reps = 10000, 5, 10
t, iqpe_trot, iqpe_iters, iqpe_max_iters = 0.2, 5, 8, 20
'''
jobs = []
for n_occ in range(spin * n_sites + 1):
    jobs.append(delayed(real_space_exact)(n_sites, t1, t2, phi, n_occ))
    jobs.append(delayed(real_space_iqpe)(n_sites, t1, t2, phi, n_occ, mapper, t, iqpe_trot, iqpe_iters))
    for _ in range(vqe_reps):
        jobs.append(delayed(real_space_vqe)(n_sites, t1, t2, phi, n_occ, mapper, vqe_iters, n_layers=vqe_layers))

results = Parallel(n_jobs=-1)(jobs)
exact, iqpe, vqe = results[0::vqe_reps+2], results[1::vqe_reps+2], [min(results[i:i+vqe_reps]) for i in range(2, len(results), vqe_reps+2)]

data = {
    "exact": {i: exact[i] for i in range(spin*n_sites+1)},
    "iqpe": {i: iqpe[i] for i in range(spin*n_sites+1)},
    "vqe": {i: vqe[i] for i in range(spin*n_sites+1)}
}
'''
file_path = os.path.join(os.getcwd(), "..", "..", "cache/haldane-model/real-space/"+str(n_sites)+"-sites/simulated-ideal-EP-t2-"+str(t2)+".json")
with open(file_path, "r") as f:
    data = json.load(f) #json.dump(data, f, indent=4)

jobs = []
for n_occ in range(spin * n_sites + 1):
    jobs.append(delayed(real_space_iqpe)(n_sites, t1, t2, phi, n_occ, mapper, t, iqpe_trot, iqpe_iters, iqpe_max_iters))

iqpe = Parallel(n_jobs=-1)(jobs)
data["iqpe"] = {i: iqpe[i] for i in range(spin*n_sites+1)}

file_path = os.path.join(os.getcwd(), "..", "..", "cache/haldane-model/real-space/"+str(n_sites)+"-sites/simulated-ideal-EP-t2-"+str(t2)+".json")
with open(file_path, "w") as f:
    json.dump(data, f, indent=4)

plt.figure()
plt.plot(range(spin*n_sites+1), data["exact"].values(), 'ro-', label="Exact")
plt.plot(range(spin*n_sites+1), data["iqpe"].values(), 'go', label=f"IQPE (t={t}, n_trot={iqpe_trot}, n_iters={iqpe_iters}, max_iters={iqpe_max_iters})")
plt.plot(range(spin*n_sites+1), data["vqe"].values(), 'bo', label=f"VQE (n_iters={vqe_iters}, n_layers={vqe_layers}, n_reps={vqe_reps})")
plt.legend()
plt.xlabel("Particle Number")
plt.ylabel("$E$")
plt.title("Real Space Haldane Hamiltonian Ground State Energy (Qiskit Aer Ideal)\n$t_1="+str(t1)+", t_2="+str(t2)+", \\phi=\\pi/"+str(int(np.pi/phi))+", N_{\\text{sites}}="+str(n_sites)+"$", fontsize=11)
plt.tight_layout()

file_path = os.path.join(os.getcwd(), "..", "..", "plots/haldane-model/real-space/"+str(n_sites)+"-sites/simulated-ideal-EP-t2-"+str(t2)+".png")
plt.savefig(file_path)