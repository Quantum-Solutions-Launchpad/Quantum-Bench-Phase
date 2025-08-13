from utils import real_space_exact, real_space_vqe, real_space_iqpe
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from qiskit_nature.second_q.mappers import JordanWignerMapper
import warnings
from joblib import Parallel, delayed

# For real_space_iqpe function: Sampler is deprecated but IQPE in Qiskit Algorithms has not been updated to use SamplerV2 yet
warnings.filterwarnings("ignore", category=DeprecationWarning)

t1, t2, phi = 1.0, 0.05, np.pi/4
spin = 2
n_sites_max = 10
mapper = JordanWignerMapper()
max_iters = 100
t, n_trot, n_iters = 0.2, 5, 8

jobs = []
for n_sites in range(3, n_sites_max+1):
    jobs.append(delayed(real_space_exact)(n_sites, t1, t2, phi, n_sites))
    jobs.append(delayed(real_space_vqe)(n_sites, t1, t2, phi, n_sites, mapper, max_iters))
    jobs.append(delayed(real_space_iqpe)(n_sites, t1, t2, phi, n_sites, mapper, t, n_trot, n_iters))

results = Parallel(n_jobs=-1)(jobs)
exact, vqe, iqpe = results[0::3], results[1::3], results[2::3]

data = {
    "exact": {i: exact[i-3] for i in range(3, n_sites_max+1)},
    "vqe": {i: vqe[i-3] for i in range(3, n_sites_max+1)},
    "iqpe": {i: iqpe[i-3] for i in range(3, n_sites_max+1)},
    "vqe_error": {i: abs(vqe[i-3]-exact[i-3]) for i in range(3, n_sites_max+1)},
    "iqpe_error": {i: abs(iqpe[i-3]-exact[i-3]) for i in range(3, n_sites_max+1)}
}

file_path = os.path.join(os.getcwd(), "..", "..", "cache/haldane-model/real-space/simulated-ideal-site-number.json")
with open(file_path, "w") as f:
    json.dump(data, f, indent=4)

plt.figure()
plt.plot(range(3, n_sites_max+1), data["exact"].values(), 'ro-', label="Exact")
plt.plot(range(3, n_sites_max+1), data["vqe"].values(), 'bo', label=f"VQE (max_iters={max_iters})")
plt.plot(range(3, n_sites_max+1), data["iqpe"].values(), 'go', label=f"IQPE (t={t}, n_trot={n_trot}, n_iters={n_iters})")
plt.legend()
plt.xlabel("Number of Sites")
plt.ylabel("$E$")
plt.title("Real Space Haldane Hamiltonian Ground State Energy (Qiskit Aer Ideal)\n$t_1="+str(t1)+", t_2="+str(t2)+", \\phi=\\pi/"+str(int(np.pi/phi))+", N_{\\text{occ}}=\\lceil N_{\\text{sites}}/2 \\rceil$", fontsize=11)
plt.tight_layout()

file_path = os.path.join(os.getcwd(), "..", "..", "plots/haldane-model/real-space/simulated-ideal-site-number.png")
plt.savefig(file_path)

x = np.arange(3, n_sites_max+1)
width = 0.25
fig, ax = plt.subplots(layout='constrained')

ax.bar(x, data["vqe_error"].values(), width, label="VQE", color="firebrick")
ax.bar(x+width, data["iqpe_error"].values(), width, label="IQPE", color="lightcoral")

ax.set_xlabel("Number of Sites")
ax.set_ylabel("Absolute Error")
ax.set_title("Real Space Haldane Hamiltonian Ground State Energy (Qiskit Aer Ideal)\n$t_1="+str(t1)+", t_2="+str(t2)+", \\phi=\\pi/"+str(int(np.pi/phi))+", N_{\\text{occ}}=\\lceil N_{\\text{sites}}/2 \\rceil$", fontsize=11)
ax.set_xticks(x+width/2, range(3, n_sites_max+1))
ax.legend()

file_path = os.path.join(os.getcwd(), "..", "..", "plots/haldane-model/real-space/simulated-ideal-site-number-error.png")
plt.savefig(file_path)