import numpy as np
import matplotlib.pyplot as plt
import os
import json
import argparse
from qiskit_nature.second_q.mappers import JordanWignerMapper
from joblib import Parallel, delayed

from core import setup_logging, real_space_vqe, real_space_iqpe, vqe_other_benchmarks, iqpe_other_benchmarks
from models import get_model

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--n-sites", type=int, default=6)
parser.add_argument("--vqe-iters", type=int, default=10000)
parser.add_argument("--vqe-layers", type=int, default=5)
parser.add_argument("--vqe-reps", type=int, default=10)
parser.add_argument("--iqpe-time", type=float, default=0.2)
parser.add_argument("--iqpe-trot", type=int, default=5)
parser.add_argument("--iqpe-iters", type=int, default=8)
parser.add_argument("--iqpe-reps", type=int, default=20)
parser.add_argument("--no-debug", action="store_true", help="Suppress debug logs")
args, _ = parser.parse_known_args()

model = get_model(args.model)
for param_name, default_val in model.DEFAULT_PARAMS.items():
    parser.add_argument(f"--{param_name}", type=type(default_val), default=default_val)
args = parser.parse_args()
model_params = {k: getattr(args, k) for k in model.DEFAULT_PARAMS}

n_sites = args.n_sites
spin = 2
mapper = JordanWignerMapper()
vqe_iters, vqe_layers, vqe_reps = args.vqe_iters, args.vqe_layers, args.vqe_reps
time_param, iqpe_trot, iqpe_iters, iqpe_reps = args.iqpe_time, args.iqpe_trot, args.iqpe_iters, args.iqpe_reps

jobs = []
stride = 3 + iqpe_reps + vqe_reps
for n_occ in range(spin * n_sites + 1):
    jobs.append(delayed(model.real_space_exact)(n_sites, n_occ, **model_params))
    for rep in range(1, iqpe_reps + 1):
        jobs.append(delayed(real_space_iqpe)(
            n_sites, n_occ, model_params,
            model.fermionic_hamiltonian,
            mapper, time_param, iqpe_trot, iqpe_iters, rep
        ))
    for rep in range(1, vqe_reps + 1):
        jobs.append(delayed(real_space_vqe)(
            n_sites, n_occ, model_params,
            model.fermionic_hamiltonian, model.get_optimizer,
            mapper, vqe_iters, vqe_layers, rep
        ))
    jobs.append(delayed(iqpe_other_benchmarks)(
        n_sites, n_occ, model_params,
        model.fermionic_hamiltonian,
        mapper, time_param, iqpe_trot, iqpe_iters, iqpe_reps
    ))
    jobs.append(delayed(vqe_other_benchmarks)(
        n_sites, n_occ, model_params,
        model.fermionic_hamiltonian,
        mapper, vqe_iters, vqe_layers, vqe_reps
    ))

def init_worker_logging():
    from core import setup_logging
    setup_logging(debug_enabled=not args.no_debug)

results = Parallel(n_jobs=-1, initializer=init_worker_logging)(jobs)

exact = results[::stride]
iqpe = [min(j for j in res if j >= -2 * n_sites - 1) for res in [results[i:i + iqpe_reps] for i in range(1, (spin * n_sites + 1) * stride, stride)]]
vqe = [min(res) for res in [results[i:i + vqe_reps] for i in range(1 + iqpe_reps, (spin * n_sites + 1) * stride, stride)]]
iqpe_other_benchmarks_results = results[1 + iqpe_reps + vqe_reps::stride]
vqe_other_benchmarks_results = results[2 + iqpe_reps + vqe_reps::stride]

logger = setup_logging(debug_enabled=not args.no_debug)
for i in range(spin * n_sites + 1):
    logger.info(f"IQPE (n_sites={n_sites}, n_occ={i}) = {iqpe[i]}")
    logger.info(f"VQE (n_sites={n_sites}, n_occ={i}) = {vqe[i]}")

data = {
    "result": {
        "exact": {i: exact[i] for i in range(spin * n_sites + 1)},
        "iqpe": {i: iqpe[i] for i in range(spin * n_sites + 1)},
        "vqe": {i: vqe[i] for i in range(spin * n_sites + 1)}
    },
    "num_queries": {
        "iqpe": {i: iqpe_other_benchmarks_results[i][0] for i in range(spin * n_sites + 1)},
        "vqe": {i: vqe_other_benchmarks_results[i][0] for i in range(spin * n_sites + 1)}
    },
    "circuit_depth": {
        "total": {
            "iqpe": {i: iqpe_other_benchmarks_results[i][1][0] for i in range(spin * n_sites + 1)},
            "vqe": {i: vqe_other_benchmarks_results[i][1][0] for i in range(spin * n_sites + 1)}
        },
        "two_qubit": {
            "iqpe": {i: iqpe_other_benchmarks_results[i][1][1] for i in range(spin * n_sites + 1)},
            "vqe": {i: vqe_other_benchmarks_results[i][1][1] for i in range(spin * n_sites + 1)}
        }
    }
}

suffix = model.file_suffix(model_params)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
cache_path = os.path.join(project_root, f"cache/{model.NAME}/{n_sites}-sites/simulated-ideal-{suffix}.json")
os.makedirs(os.path.dirname(cache_path), exist_ok=True)
with open(cache_path, "w") as f:
    json.dump(data, f, indent=4)

param_str = ", ".join(f"${label}={model_params[k]}$" for k, label in model.PARAM_LABELS.items())
title = f"Real Space {model.DISPLAY_NAME} Hamiltonian Ground State Energy (Qiskit Aer Ideal)\n{param_str}, $N_{{\\text{{sites}}}}={n_sites}$"

plt.figure()
plt.plot(range(spin * n_sites + 1), data["result"]["exact"].values(), 'ro-', label="Exact")
plt.plot(range(spin * n_sites + 1), data["result"]["iqpe"].values(), 'go', label=f"IQPE (t={time_param}, n_trot={iqpe_trot}, n_iters={iqpe_iters}, n_reps={iqpe_reps})")
plt.plot(range(spin * n_sites + 1), data["result"]["vqe"].values(), 'bo', label=f"VQE (n_iters={vqe_iters}, n_layers={vqe_layers}, n_reps={vqe_reps})")
plt.legend()
plt.xlabel("Particle Number")
plt.ylabel("$E$")
plt.title(title, fontsize=11)
plt.grid(True)
plt.tight_layout()

plot_path = os.path.join(project_root, f"plots/{model.NAME}/{n_sites}-sites/simulated-ideal-{suffix}.png")
os.makedirs(os.path.dirname(plot_path), exist_ok=True)
plt.savefig(plot_path)
