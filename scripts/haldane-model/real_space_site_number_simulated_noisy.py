from utils import real_space_exact, real_space_vqe, real_space_iqpe
import numpy as np
import matplotlib.pyplot as plt
import os
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_ibm_runtime.fake_provider import FakeTorino
import warnings

# For real_space_iqpe function: Sampler is deprecated but IQPE in Qiskit Algorithms has not been updated to use SamplerV2 yet
warnings.filterwarnings("ignore", category=DeprecationWarning)

t1, t2, phi = 1.0, 0.05, np.pi/4
spin = 2
n_sites_max = 10
mapper = JordanWignerMapper()
max_iters = 20
t, n_trot, n_iters = 0.2, 3, 6
backend = FakeTorino()

data = {
    "exact": {},
    "vqe": {},
    "iqpe": {}
}
for n_sites in range(3, n_sites_max+1):
    data["exact"][n_sites] = real_space_exact(n_sites, t1, t2, phi, n_sites)
    data["vqe"][n_sites] = real_space_vqe(n_sites, t1, t2, phi, n_sites, mapper, max_iters, backend)
    data["iqpe"][n_sites] = real_space_iqpe(n_sites, t1, t2, phi, n_sites, mapper, t, n_trot, n_iters, backend)

plt.figure()
plt.plot(range(3, n_sites_max+1), data["exact"].values(), 'ro-', label="Exact")
plt.plot(range(3, n_sites_max+1), data["vqe"].values(), 'bo', label=f"VQE (max_iters={max_iters})")
plt.plot(range(3, n_sites_max+1), data["iqpe"].values(), 'go', label=f"IQPE (t={t}, n_trot={n_trot}, n_iters={n_iters})")
plt.legend()
plt.xlabel("Number of Sites")
plt.ylabel("$E$")
plt.title("Real Space Haldane Hamiltonian Ground State Energy (Qiskit Aer Noisy)", fontsize=11)
plt.tight_layout()

file_path = os.path.join(os.getcwd(), "..", "..", "plots/haldane-model/real-space/simulated-noisy-site-number.png")
plt.savefig(file_path)