from utils import real_space_exact, real_space_vqe, real_space_iqpe
import numpy as np
import matplotlib.pyplot as plt
import os
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_ibm_runtime.fake_provider import FakeTorino
import warnings

# For real_space_iqpe function: Sampler is deprecated but IQPE in Qiskit Algorithms has not been updated to use SamplerV2 yet
warnings.filterwarnings("ignore", category=DeprecationWarning)

n_sites = 6
t1, t2, phi = 1.0, 0.05, np.pi/4
spin = 2
mapper = JordanWignerMapper()
max_iters = 20
t, n_trot, n_iters = 0.2, 3, 6
backend = FakeTorino()

data = {
    "exact": {},
    "vqe": {},
    "iqpe": {}
}
for n_occ in range(spin*n_sites+1):
    data["exact"][n_occ] = real_space_exact(n_sites, t1, t2, phi, n_occ)
    data["vqe"][n_occ] = real_space_vqe(n_sites, t1, t2, phi, n_occ, mapper, max_iters, backend)
    data["iqpe"][n_occ] = real_space_iqpe(n_sites, t1, t2, phi, n_occ, mapper, t, n_trot, n_iters, backend)

plt.figure()
plt.plot(range(spin*n_sites+1), data["exact"].values(), 'ro-', label="Exact")
plt.plot(range(spin*n_sites+1), data["vqe"].values(), 'bo', label=f"VQE (max_iters={max_iters})")
plt.plot(range(spin*n_sites+1), data["iqpe"].values(), 'go', label=f"IQPE (t={t}, n_trot={n_trot}, n_iters={n_iters})")
plt.legend()
plt.xlabel("Particle Number")
plt.ylabel("$E$")
plt.title("Real Space Haldane Hamiltonian Ground State Energy (Qiskit Aer Noisy)", fontsize=11)
plt.tight_layout()

file_path = os.path.join(os.getcwd(), "..", "..", "plots/haldane-model/real-space/simulated-noisy-particle-number-"+str(n_sites)+"-sites.png")
plt.savefig(file_path)