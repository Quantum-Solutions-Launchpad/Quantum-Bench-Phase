from utils import hubbard_real_space_exact, hubbard_real_space_iqpe, hubbard_real_space_vqe, hubbard_iqpe_other_benchmarks, hubbard_vqe_other_benchmarks, setup_logging
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from qiskit_nature.second_q.mappers import JordanWignerMapper
from joblib import Parallel, delayed
import argparse

n_sites = 4
t, U = 1.0, 0.0
spin = 2
mapper = JordanWignerMapper()
vqe_iters, vqe_layers, vqe_reps = 10000, 5, 10
time_param, iqpe_trot, iqpe_iters, iqpe_reps = 0.2, 5, 3, 5

parser = argparse.ArgumentParser()
parser.add_argument("--no-debug", action="store_true", help="Suppress debug logs")
args = parser.parse_args()

jobs = []
for n_occ in range(spin * n_sites + 1):
    jobs.append(delayed(hubbard_real_space_exact)(n_sites, t, U, n_occ))
    # for rep in range(1, iqpe_reps+1):
    #     jobs.append(delayed(hubbard_real_space_iqpe)(n_sites, t, U, n_occ, mapper, time_param, iqpe_trot, iqpe_iters, rep))
    for rep in range(1, vqe_reps+1):
        jobs.append(delayed(hubbard_real_space_vqe)(n_sites, t, U, n_occ, mapper, vqe_iters, vqe_layers, rep))
    # jobs.append(delayed(hubbard_iqpe_other_benchmarks)(n_sites, t, U, n_occ, mapper, time_param, iqpe_trot, iqpe_iters, iqpe_reps))
    jobs.append(delayed(hubbard_vqe_other_benchmarks)(n_sites, t, U, n_occ, mapper, vqe_iters, vqe_layers, vqe_reps))

def init_worker_logging():
    from utils import setup_logging
    setup_logging(debug_enabled=not args.no_debug)

results = Parallel(n_jobs=-1, initializer=init_worker_logging)(jobs)

exact = results[::2+vqe_reps]
# iqpe = [min(j for j in res if j >= -2*n_sites-1) for res in [results[i:i+iqpe_reps] for i in range(1, (spin*n_sites+1)*(3+iqpe_reps+vqe_reps), 3+iqpe_reps+vqe_reps)]]
vqe = [min(res) for res in [results[i:i+vqe_reps] for i in range(1, (spin*n_sites+1)*(2+vqe_reps), 2+vqe_reps)]]
# iqpe_other_benchmarks = results[1+iqpe_reps+vqe_reps::3+iqpe_reps+vqe_reps]
vqe_other_benchmarks = results[1+vqe_reps::2+vqe_reps]

logger = setup_logging(debug_enabled=not args.no_debug)
for i in range(spin*n_sites+1):
    # logger.info(f"IQPE (n_sites={n_sites}, n_occ={i}) = {iqpe[i]}")
    logger.info(f"VQE (n_sites={n_sites}, n_occ={i}) = {vqe[i]}")

data = {
    "result": {
        "exact": {i: exact[i] for i in range(spin*n_sites+1)},
        # "iqpe": {i: iqpe[i] for i in range(spin*n_sites+1)},
        "vqe": {i: vqe[i] for i in range(spin*n_sites+1)}
    },
    "num_queries": {
        # "iqpe": {i: iqpe_other_benchmarks[i][0] for i in range(spin*n_sites+1)},
        "vqe": {i: vqe_other_benchmarks[i][0] for i in range(spin*n_sites+1)}
    },
    "circuit_depth": {
        "total": {
            # "iqpe": {i: iqpe_other_benchmarks[i][1][0] for i in range(spin*n_sites+1)},
            "vqe": {i: vqe_other_benchmarks[i][1][0] for i in range(spin*n_sites+1)}
        },
        "two_qubit": {
            # "iqpe": {i: iqpe_other_benchmarks[i][1][1] for i in range(spin*n_sites+1)},
            "vqe": {i: vqe_other_benchmarks[i][1][1] for i in range(spin*n_sites+1)}
        }
    }
}

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
file_path = os.path.join(project_root, "cache/hubbard-model/real-space/"+str(n_sites)+"-sites/simulated-ideal-U-"+str(U)+".json")
os.makedirs(os.path.dirname(file_path), exist_ok=True)
with open(file_path, "w") as f:
    json.dump(data, f, indent=4)

plt.figure()
plt.plot(range(spin*n_sites+1), data["result"]["exact"].values(), 'ro-', label="Exact")
# plt.plot(range(spin*n_sites+1), data["result"]["iqpe"].values(), 'go', label=f"IQPE (t={time_param}, n_trot={iqpe_trot}, n_iters={iqpe_iters}, n_reps={iqpe_reps})")
plt.plot(range(spin*n_sites+1), data["result"]["vqe"].values(), 'bo', label=f"VQE (n_iters={vqe_iters}, n_layers={vqe_layers}, n_reps={vqe_reps})")
plt.legend()
plt.xlabel("Particle Number")
plt.ylabel("$E$")
plt.title("Real Space Hubbard Hamiltonian Ground State Energy (Qiskit Aer Ideal)\n$t="+str(t)+", U="+str(U)+", N_{\\text{sites}}="+str(n_sites)+"$", fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()

file_path = os.path.join(project_root, "plots/hubbard-model/"+str(n_sites)+"-sites/simulated-ideal-U-"+str(U)+".png")
os.makedirs(os.path.dirname(file_path), exist_ok=True)
plt.savefig(file_path)
