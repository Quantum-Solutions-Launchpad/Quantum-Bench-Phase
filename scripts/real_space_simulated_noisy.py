import numpy as np
import matplotlib.pyplot as plt
import os
import json
import argparse
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
from joblib import Parallel, delayed

from core import setup_logging, real_space_vqe, real_space_iqpe, vqe_other_benchmarks, iqpe_other_benchmarks
from models import get_model

backend = FakeSherbrooke()

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

def tagged_job(tag, func, *args, **kwargs):
    return tag, func(*args, **kwargs)

jobs = []
for n_occ in range(spin * n_sites + 1):
    jobs.append(delayed(tagged_job)(("exact", n_occ), model.real_space_exact, n_sites, n_occ, **model_params))
    for rep in range(1, iqpe_reps + 1):
        jobs.append(delayed(tagged_job)(
            ("iqpe", n_occ, rep), real_space_iqpe,
            n_sites, n_occ, model_params, model.fermionic_hamiltonian,
            mapper, time_param, iqpe_trot, iqpe_iters, rep,
            backend=backend
        ))
    for rep in range(1, vqe_reps + 1):
        jobs.append(delayed(tagged_job)(
            ("vqe", n_occ, rep), real_space_vqe,
            n_sites, n_occ, model_params, model.fermionic_hamiltonian, model.get_optimizer,
            mapper, vqe_iters, vqe_layers, rep,
            backend=backend
        ))
    jobs.append(delayed(tagged_job)(
        ("iqpe_bench", n_occ), iqpe_other_benchmarks,
        n_sites, n_occ, model_params, model.fermionic_hamiltonian,
        mapper, time_param, iqpe_trot, iqpe_iters, iqpe_reps,
        backend=backend
    ))
    jobs.append(delayed(tagged_job)(
        ("vqe_bench", n_occ), vqe_other_benchmarks,
        n_sites, n_occ, model_params, model.fermionic_hamiltonian,
        mapper, vqe_iters, vqe_layers, vqe_reps,
        backend=backend
    ))

suffix = model.file_suffix(model_params)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
raw_data_path = os.path.join(project_root, f"logs/{model.NAME}/{n_sites}-sites/raw-data/simulated-noisy-{suffix}.json")
os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)

n_occ_count = spin * n_sites + 1
raw_data = {
    "parameters": {
        "model": model.NAME,
        "n_sites": n_sites,
        "simulation": "noisy",
        "model_params": {k: float(v) for k, v in model_params.items()},
        "vqe": {"iters": vqe_iters, "layers": vqe_layers, "reps": vqe_reps},
        "iqpe": {"time": time_param, "trot": iqpe_trot, "iters": iqpe_iters, "reps": iqpe_reps}
    },
    "occupations": {
        str(i): {
            "exact": None,
            "vqe": {"repetitions": [], "num_queries": None, "circuit_depth": None},
            "iqpe": {"repetitions": [], "iteration_energies": [], "num_queries": None, "circuit_depth": None}
        }
        for i in range(n_occ_count)
    }
}

with open(raw_data_path, "w") as f:
    json.dump(raw_data, f, indent=4)

def init_worker_logging():
    from core import setup_logging
    setup_logging(debug_enabled=not args.no_debug)

for tag, result in Parallel(n_jobs=-1, return_as="generator_unordered", initializer=init_worker_logging)(jobs):
    occ = str(tag[1])
    if tag[0] == "exact":
        raw_data["occupations"][occ]["exact"] = result
    elif tag[0] == "iqpe":
        energy, iter_energies = result
        raw_data["occupations"][occ]["iqpe"]["repetitions"].append(energy)
        raw_data["occupations"][occ]["iqpe"]["iteration_energies"].append(iter_energies)
    elif tag[0] == "vqe":
        raw_data["occupations"][occ]["vqe"]["repetitions"].append(result)
    elif tag[0] == "iqpe_bench":
        num_q, (total, two_q) = result
        raw_data["occupations"][occ]["iqpe"]["num_queries"] = num_q
        raw_data["occupations"][occ]["iqpe"]["circuit_depth"] = {"total": total, "two_qubit": two_q}
    elif tag[0] == "vqe_bench":
        num_q, (total, two_q) = result
        raw_data["occupations"][occ]["vqe"]["num_queries"] = num_q
        raw_data["occupations"][occ]["vqe"]["circuit_depth"] = {"total": total, "two_qubit": two_q}
    with open(raw_data_path, "w") as f:
        json.dump(raw_data, f, indent=4)

logger = setup_logging(debug_enabled=not args.no_debug)

exact = [raw_data["occupations"][str(i)]["exact"] for i in range(n_occ_count)]
iqpe_reps_data = [raw_data["occupations"][str(i)]["iqpe"]["repetitions"] for i in range(n_occ_count)]
iqpe = [min(reps) for reps in iqpe_reps_data]
vqe = [min(raw_data["occupations"][str(i)]["vqe"]["repetitions"]) for i in range(n_occ_count)]

for i in range(n_occ_count):
    logger.info(f"IQPE (n_sites={n_sites}, n_occ={i}) = {iqpe[i]}")
    logger.info(f"VQE (n_sites={n_sites}, n_occ={i}) = {vqe[i]}")

data = {
    "result": {
        "exact": {i: exact[i] for i in range(n_occ_count)},
        "iqpe": {i: iqpe[i] for i in range(n_occ_count)},
        "vqe": {i: vqe[i] for i in range(n_occ_count)}
    },
    "num_queries": {
        "iqpe": {i: raw_data["occupations"][str(i)]["iqpe"]["num_queries"] for i in range(n_occ_count)},
        "vqe": {i: raw_data["occupations"][str(i)]["vqe"]["num_queries"] for i in range(n_occ_count)}
    },
    "circuit_depth": {
        "total": {
            "iqpe": {i: raw_data["occupations"][str(i)]["iqpe"]["circuit_depth"]["total"] for i in range(n_occ_count)},
            "vqe": {i: raw_data["occupations"][str(i)]["vqe"]["circuit_depth"]["total"] for i in range(n_occ_count)}
        },
        "two_qubit": {
            "iqpe": {i: raw_data["occupations"][str(i)]["iqpe"]["circuit_depth"]["two_qubit"] for i in range(n_occ_count)},
            "vqe": {i: raw_data["occupations"][str(i)]["vqe"]["circuit_depth"]["two_qubit"] for i in range(n_occ_count)}
        }
    }
}

log_path = os.path.join(project_root, f"logs/{model.NAME}/{n_sites}-sites/simulated-noisy-{suffix}.json")
os.makedirs(os.path.dirname(log_path), exist_ok=True)
with open(log_path, "w") as f:
    json.dump(data, f, indent=4)

param_str = ", ".join(f"${label}={model_params[k]}$" for k, label in model.PARAM_LABELS.items())
title = f"Real Space {model.DISPLAY_NAME} Hamiltonian Ground State Energy (Qiskit Aer Noisy)\n{param_str}, $N_{{\\text{{sites}}}}={n_sites}$"

plt.figure()
plt.plot(range(n_occ_count), data["result"]["exact"].values(), 'ro-', label="Exact")
plt.plot(range(n_occ_count), data["result"]["iqpe"].values(), 'go', label=f"IQPE (t={time_param}, n_trot={iqpe_trot}, n_iters={iqpe_iters}, n_reps={iqpe_reps})")
plt.plot(range(n_occ_count), data["result"]["vqe"].values(), 'bo', label=f"VQE (n_iters={vqe_iters}, n_layers={vqe_layers}, n_reps={vqe_reps})")
plt.legend()
plt.xlabel("Particle Number")
plt.ylabel("$E$")
plt.title(title, fontsize=11)
plt.grid(True)
plt.tight_layout()

plot_path = os.path.join(project_root, f"plots/{model.NAME}/{n_sites}-sites/simulated-noisy-{suffix}.png")
os.makedirs(os.path.dirname(plot_path), exist_ok=True)
plt.savefig(plot_path)
