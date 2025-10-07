from utils import real_space_exact, real_space_iqpe, real_space_vqe
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from qiskit_nature.second_q.mappers import JordanWignerMapper
from joblib import Parallel, delayed
import argparse

n_sites = 4
t1, t2, phi = 1.0, 0.0, np.pi/4
spin = 2
mapper = JordanWignerMapper()
vqe_iters, vqe_layers, vqe_reps = 10000, 5, 10
t, iqpe_trot, iqpe_iters, iqpe_reps = 0.2, 5, 8, 20

parser = argparse.ArgumentParser()
parser.add_argument("--no-debug", action="store_true", help="Suppress debug logs")
args = parser.parse_args()

jobs = []
for n_occ in range(spin * n_sites + 1):
    jobs.append(delayed(real_space_exact)(n_sites, t1, t2, phi, n_occ))
    jobs.append(delayed(real_space_iqpe)(n_sites, t1, t2, phi, n_occ, mapper, t, iqpe_trot, iqpe_iters, iqpe_reps))
    jobs.append(delayed(real_space_vqe)(n_sites, t1, t2, phi, n_occ, mapper, vqe_iters, vqe_layers, vqe_reps))

def init_worker_logging():
    from utils import setup_logging
    setup_logging(debug_enabled=not args.no_debug)

results = Parallel(n_jobs=-1, initializer=init_worker_logging)(jobs)
exact, iqpe, vqe = results[0::3], results[1::3], results[2::3]

data = {
    "result": {
        "exact": {i: exact[i] for i in range(spin*n_sites+1)},
        "iqpe": {i: iqpe["result"][i] for i in range(spin*n_sites+1)},
        "vqe": {i: vqe["result"][i] for i in range(spin*n_sites+1)}
    },
    "num_queries": {
        "iqpe": {i: iqpe["num_queries"][i] for i in range(spin*n_sites+1)},
        "vqe": {i: vqe["num_queries"][i] for i in range(spin*n_sites+1)}
    },
    "circuit_depth": {
        "total": {
            "iqpe": {i: iqpe["circuit_depth"][i][0] for i in range(spin*n_sites+1)},
            "vqe": {i: vqe["circuit_depth"][i][0] for i in range(spin*n_sites+1)}
        },
        "two_qubit": {
            "iqpe": {i: iqpe["circuit_depth"][i][1] for i in range(spin*n_sites+1)},
            "vqe": {i: vqe["circuit_depth"][i][1] for i in range(spin*n_sites+1)}
        }
    }
}

file_path = os.path.join(os.getcwd(), "..", "..", "cache/haldane-model/real-space/"+str(n_sites)+"-sites/simulated-ideal-t2-"+str(t2)+".json")
with open(file_path, "w") as f:
    json.dump(data, f, indent=4)

plt.figure()
plt.plot(range(spin*n_sites+1), data["result"]["exact"].values(), 'ro-', label="Exact")
plt.plot(range(spin*n_sites+1), data["result"]["iqpe"].values(), 'go', label=f"IQPE (t={t}, n_trot={iqpe_trot}, n_iters={iqpe_iters}, n_reps={iqpe_reps})")
plt.plot(range(spin*n_sites+1), data["result"]["vqe"].values(), 'bo', label=f"VQE (n_iters={vqe_iters}, n_layers={vqe_layers}, n_reps={vqe_reps})")
plt.legend()
plt.xlabel("Particle Number")
plt.ylabel("$E$")
plt.title("Real Space Haldane Hamiltonian Ground State Energy (Qiskit Aer Ideal)\n$t_1="+str(t1)+", t_2="+str(t2)+", \\phi=\\pi/"+str(int(np.pi/phi))+", N_{\\text{sites}}="+str(n_sites)+"$", fontsize=11)
plt.tight_layout()

file_path = os.path.join(os.getcwd(), "..", "..", "plots/haldane-model/real-space/"+str(n_sites)+"-sites/simulated-ideal-t2-"+str(t2)+".png")
plt.savefig(file_path)