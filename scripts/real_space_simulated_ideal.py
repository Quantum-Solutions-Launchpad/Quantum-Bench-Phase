import numpy as np
import matplotlib.pyplot as plt
import os
import json
import argparse
from datetime import datetime
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
parser.add_argument("--task-index", type=int, default=None, help="Run only this zero-based shard of the job list")
parser.add_argument("--task-count", type=int, default=1, help="Total number of job-list shards")
parser.add_argument("--prepare-only", action="store_true", help="Initialize output/progress files and exit")
parser.add_argument("--aggregate-only", action="store_true", help="Build final outputs from the progress JSONL and exit")
parser.add_argument("--no-progress-log", action="store_true", help="Disable append-only progress logging")
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
            mapper, time_param, iqpe_trot, iqpe_iters, rep
        ))
    for rep in range(1, vqe_reps + 1):
        jobs.append(delayed(tagged_job)(
            ("vqe", n_occ, rep), real_space_vqe,
            n_sites, n_occ, model_params, model.fermionic_hamiltonian, model.get_optimizer,
            mapper, vqe_iters, vqe_layers, rep
        ))
    jobs.append(delayed(tagged_job)(
        ("iqpe_bench", n_occ), iqpe_other_benchmarks,
        n_sites, n_occ, model_params, model.fermionic_hamiltonian,
        mapper, time_param, iqpe_trot, iqpe_iters, iqpe_reps
    ))
    jobs.append(delayed(tagged_job)(
        ("vqe_bench", n_occ), vqe_other_benchmarks,
        n_sites, n_occ, model_params, model.fermionic_hamiltonian,
        mapper, vqe_iters, vqe_layers, vqe_reps
    ))

suffix = model.file_suffix(model_params)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
raw_data_path = os.path.join(project_root, f"logs/{model.NAME}/{n_sites}-sites/raw-data/simulated-ideal-{suffix}.json")
progress_path = os.path.join(project_root, f"logs/{model.NAME}/{n_sites}-sites/raw-data/simulated-ideal-{suffix}.progress.jsonl")
os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)

n_occ_count = spin * n_sites + 1
raw_data = {
    "parameters": {
        "model": model.NAME,
        "n_sites": n_sites,
        "simulation": "ideal",
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

def init_worker_logging():
    from core import setup_logging
    setup_logging(debug_enabled=not args.no_debug)

def append_progress(tag, result):
    """Append one completed job result to the progress JSONL file."""
    if args.no_progress_log:
        return
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "tag": list(tag),
        "result": result,
    }
    payload = (json.dumps(record) + "\n").encode()
    fd = os.open(progress_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND)
    try:
        os.write(fd, payload)
    finally:
        os.close(fd)

def apply_result(raw_data, tag, result):
    """Insert one tagged result into the in-memory raw data structure."""
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

def load_progress(raw_data):
    """Rebuild raw_data by replaying every result in the progress JSONL file."""
    if not os.path.exists(progress_path):
        raise FileNotFoundError(f"Progress file does not exist: {progress_path}")
    with open(progress_path, "r") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            apply_result(raw_data, tuple(record["tag"]), record["result"])
    return raw_data

def validate_complete(raw_data):
    """Check that every exact, IQPE, VQE, and benchmark result is present."""
    missing = []
    for i in range(n_occ_count):
        occ = str(i)
        entry = raw_data["occupations"][occ]
        if entry["exact"] is None:
            missing.append(f"exact:{i}")
        if len(entry["iqpe"]["repetitions"]) != iqpe_reps:
            missing.append(f"iqpe:{i} ({len(entry['iqpe']['repetitions'])}/{iqpe_reps})")
        if len(entry["vqe"]["repetitions"]) != vqe_reps:
            missing.append(f"vqe:{i} ({len(entry['vqe']['repetitions'])}/{vqe_reps})")
        if entry["iqpe"]["num_queries"] is None or entry["iqpe"]["circuit_depth"] is None:
            missing.append(f"iqpe_bench:{i}")
        if entry["vqe"]["num_queries"] is None or entry["vqe"]["circuit_depth"] is None:
            missing.append(f"vqe_bench:{i}")
    if missing:
        raise RuntimeError("Missing results before aggregation: " + ", ".join(missing[:20]))

def write_outputs(raw_data):
    """Write completed raw data, summary JSON, and the final plot."""
    validate_complete(raw_data)

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

    log_path = os.path.join(project_root, f"logs/{model.NAME}/{n_sites}-sites/simulated-ideal-{suffix}.json")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(data, f, indent=4)

    param_str = ", ".join(f"${label}={model_params[k]}$" for k, label in model.PARAM_LABELS.items())
    title = f"Real Space {model.DISPLAY_NAME} Hamiltonian Ground State Energy (Qiskit Aer Ideal)\n{param_str}, $N_{{\\text{{sites}}}}={n_sites}$"

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

    plot_path = os.path.join(project_root, f"plots/{model.NAME}/{n_sites}-sites/simulated-ideal-{suffix}.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)

# Check that shard arguments are valid.
if args.task_count < 1:
    raise ValueError("--task-count must be at least 1")
if args.task_index is not None and not 0 <= args.task_index < args.task_count:
    raise ValueError("--task-index must satisfy 0 <= task-index < task-count")

# Prepare the raw/progress files before we begin slurm shard tasks.
if args.prepare_only:
    if not args.no_progress_log:
        with open(progress_path, "w") as f:
            pass
    with open(raw_data_path, "w") as f:
        json.dump(raw_data, f, indent=4)
    raise SystemExit(0)

# Rebuild final outputs after all Slurm shard tasks finish.
if args.aggregate_only:
    write_outputs(load_progress(raw_data))
    raise SystemExit(0)

# Run the full job list in one process, or only this shard's subset.
if args.task_index is None: # Full unsharded run
    if not args.no_progress_log:
        with open(progress_path, "w") as f:
            pass
    job_results = Parallel(n_jobs=-1, return_as="generator_unordered", initializer=init_worker_logging)(jobs)
else: # One Slurm shard
    init_worker_logging()
    shard_jobs = [job for idx, job in enumerate(jobs) if idx % args.task_count == args.task_index]
    job_results = Parallel(n_jobs=1, return_as="generator_unordered")(shard_jobs)

# Save each result immediately so completed shard work survives process exits.
for tag, result in job_results:
    append_progress(tag, result) #add shard result to progress file
    apply_result(raw_data, tag, result) #mutatae the Python dict raw_data in memory (doesn't write the file yet)

# Shard tasks only append progress; final aggregation happens separately.
if args.task_index is not None:
    raise SystemExit(0)
write_outputs(raw_data)
