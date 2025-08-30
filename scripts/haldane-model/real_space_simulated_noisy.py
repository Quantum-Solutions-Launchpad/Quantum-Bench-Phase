from utils import real_space_exact, real_space_vqe, real_space_iqpe
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from qiskit_nature.second_q.mappers import JordanWignerMapper
import warnings
from joblib import Parallel, delayed
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke

# For real_space_iqpe function: Sampler is deprecated but IQPE in Qiskit Algorithms has not been updated to use SamplerV2 yet
warnings.filterwarnings("ignore", category=DeprecationWarning)

n_sites = 6
t1, t2, phi = 1.0, 0.05, np.pi/4
spin = 2
mapper = JordanWignerMapper()
max_iters = 20
t, n_trot, n_iters = 0.2, 5, 8
backend = FakeSherbrooke()

jobs = []
for n_occ in range(spin * n_sites + 1):
    jobs.append(delayed(real_space_exact)(n_sites, t1, t2, phi, n_occ))
    jobs.append(delayed(real_space_vqe)(n_sites, t1, t2, phi, n_occ, mapper, max_iters, backend))
    jobs.append(delayed(real_space_iqpe)(n_sites, t1, t2, phi, n_occ, mapper, t, n_trot, n_iters, backend))

results = Parallel(n_jobs=-1)(jobs)
exact, vqe, iqpe = results[0::3], results[1::3], results[2::3]

data = {
    "exact": {i: exact[i] for i in range(spin*n_sites+1)},
    "vqe": {i: vqe[i] for i in range(spin*n_sites+1)},
    "iqpe": {i: iqpe[i] for i in range(spin*n_sites+1)},
    "vqe_error": {i: abs(vqe[i]-exact[i]) for i in range(spin*n_sites+1)},
    "iqpe_error": {i: abs(iqpe[i]-exact[i]) for i in range(spin*n_sites+1)}
}

file_path = os.path.join(os.getcwd(), "..", "..", "cache/haldane-model/real-space/"+str(n_sites)+"-sites/simulated-noisy.json")
with open(file_path, "w") as f:
    json.dump(data, f, indent=4)

plt.figure()
plt.plot(range(spin*n_sites+1), data["exact"].values(), 'ro-', label="Exact")
plt.plot(range(spin*n_sites+1), data["vqe"].values(), 'bo', label=f"VQE (max_iters={max_iters})")
plt.plot(range(spin*n_sites+1), data["iqpe"].values(), 'go', label=f"IQPE (t={t}, n_trot={n_trot}, n_iters={n_iters})")
plt.legend()
plt.xlabel("Particle Number")
plt.ylabel("$E$")
plt.title("Real Space Haldane Hamiltonian Ground State Energy (Qiskit Aer Noisy)\n$t_1="+str(t1)+", t_2="+str(t2)+", \\phi=\\pi/"+str(int(np.pi/phi))+", N_{\\text{sites}}="+str(n_sites)+"$", fontsize=11)
plt.tight_layout()

file_path = os.path.join(os.getcwd(), "..", "..", "plots/haldane-model/real-space/"+str(n_sites)+"-sites/simulated-noisy.png")
plt.savefig(file_path)

x = np.arange(spin*n_sites+1)
width = 0.25
fig, ax = plt.subplots(layout='constrained')

ax.bar(x, data["vqe_error"].values(), width, label="VQE", color="firebrick")
ax.bar(x+width, data["iqpe_error"].values(), width, label="IQPE", color="lightcoral")

ax.set_xlabel("Particle Number")
ax.set_ylabel("Absolute Error")
ax.set_title("Real Space Haldane Hamiltonian Ground State Energy (Qiskit Aer Noisy)\n$t_1="+str(t1)+", t_2="+str(t2)+", \\phi=\\pi/"+str(int(np.pi/phi))+", N_{\\text{sites}}="+str(n_sites)+"$", fontsize=11)
ax.set_xticks(x+width/2, range(spin*n_sites+1))
ax.legend()

file_path = os.path.join(os.getcwd(), "..", "..", "plots/haldane-model/real-space/"+str(n_sites)+"-sites/simulated-noisy-error.png")
plt.savefig(file_path)