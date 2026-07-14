import argparse
import json
import math
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from qiskit_nature.second_q.mappers import JordanWignerMapper


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from qbp._core import analytic, iqpe, iqpe_other_benchmarks, setup_logging, vqe, vqe_other_benchmarks
from qbp._registry import get_model


def _model_param_names(model):
    return list(model.param_labels)


def _default_param_value(name):
    if name in ("t", "t1"):
        return 1.0
    if name == "phi":
        return math.pi / 4
    return 0.0


def _legacy_lattice(model, n_sites):
    if n_sites % model.sites_per_cell != 0:
        raise ValueError(
            f"n-sites={n_sites} is not divisible by sites_per_cell={model.sites_per_cell} "
            f"for model '{model.name}'."
        )
    unit_cells = n_sites // model.sites_per_cell
    if model.n_dims == 1:
        return (unit_cells,)
    return (unit_cells,) + (1,) * (model.n_dims - 1)


def _file_suffix(model_name, params):
    if model_name == "haldane":
        return f"t2-{params['t2']}"
    if model_name == "hubbard":
        return f"U-{params['U']}"
    if model_name == "haldane-hubbard":
        return f"U-{params['U']}-t2-{params['t2']}"
    parts = []
    for key in sorted(params):
        parts.append(f"{key}-{params[key]}")
    return "-".join(parts)


def main(simulation_tag, backend=None, argv=None):
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
    parser.add_argument("--task-index", type=int, default=None)
    parser.add_argument("--task-count", type=int, default=1)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--no-progress-log", action="store_true")
    parser.add_argument("--no-debug", action="store_true")
    args, _ = parser.parse_known_args(argv)

    model = get_model(args.model)
    for param_name in _model_param_names(model):
        default_val = _default_param_value(param_name)
        parser.add_argument(f"--{param_name}", type=type(default_val), default=default_val)
    args = parser.parse_args(argv)

    model_params = {k: getattr(args, k) for k in _model_param_names(model)}
    n_sites = args.n_sites
    lattice = _legacy_lattice(model, n_sites)
    spin = model.spin
    mapper = model.get_mapper(n_sites, spin, n_sites * spin // 2) if hasattr(model, "get_mapper") else JordanWignerMapper()
    vqe_iters, vqe_layers, vqe_reps = args.vqe_iters, args.vqe_layers, args.vqe_reps
    time_param, iqpe_trot, iqpe_iters, iqpe_reps = args.iqpe_time, args.iqpe_trot, args.iqpe_iters, args.iqpe_reps

    def tagged_job(tag, func, *job_args, **job_kwargs):
        return tag, func(*job_args, **job_kwargs)

    def maybe_backend(job_kwargs):
        if backend is not None:
            job_kwargs["backend"] = backend
        return job_kwargs

    jobs = []
    for n_occ in range(spin * n_sites + 1):
        mapper = model.get_mapper(n_sites, spin, n_occ) if hasattr(model, "get_mapper") else JordanWignerMapper()
        jobs.append(delayed(tagged_job)(("exact", n_occ), analytic, model, lattice, n_occ, model_params))
        for rep in range(1, iqpe_reps + 1):
            jobs.append(delayed(tagged_job)(
                ("iqpe", n_occ, rep), iqpe,
                lattice, n_sites, spin, n_occ, model_params, model.fermionic_hamiltonian,
                mapper, time_param, iqpe_trot, iqpe_iters, rep,
                **maybe_backend({})
            ))
        for rep in range(1, vqe_reps + 1):
            jobs.append(delayed(tagged_job)(
                ("vqe", n_occ, rep), vqe,
                lattice, n_sites, spin, n_occ, model_params, model.fermionic_hamiltonian, model.get_optimizer,
                model.get_vqe_ansatz, mapper, vqe_iters, vqe_layers, rep,
                **maybe_backend({})
            ))
        jobs.append(delayed(tagged_job)(
            ("iqpe_bench", n_occ), iqpe_other_benchmarks,
            lattice, n_sites, spin, n_occ, model_params, model.fermionic_hamiltonian,
            mapper, time_param, iqpe_trot, iqpe_iters, iqpe_reps,
            **maybe_backend({})
        ))
        jobs.append(delayed(tagged_job)(
            ("vqe_bench", n_occ), vqe_other_benchmarks,
            lattice, n_sites, spin, n_occ, model_params, model.fermionic_hamiltonian,
            model.get_vqe_ansatz, mapper, vqe_iters, vqe_layers, vqe_reps,
            **maybe_backend({})
        ))

    suffix = _file_suffix(model.NAME, model_params)
    raw_data_path = os.path.join(REPO_ROOT, f"logs/{model.NAME}/{n_sites}-sites/raw-data/simulated-{simulation_tag}-{suffix}.json")
    progress_path = os.path.join(REPO_ROOT, f"logs/{model.NAME}/{n_sites}-sites/raw-data/simulated-{simulation_tag}-{suffix}.progress.jsonl")
    os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)

    n_occ_count = spin * n_sites + 1
    raw_data = {
        "parameters": {
            "model": model.NAME,
            "n_sites": n_sites,
            "simulation": simulation_tag,
            "model_params": {k: float(v) for k, v in model_params.items()},
            "vqe": {"iters": vqe_iters, "layers": vqe_layers, "reps": vqe_reps},
            "iqpe": {"time": time_param, "trot": iqpe_trot, "iters": iqpe_iters, "reps": iqpe_reps},
        },
        "occupations": {
            str(i): {
                "exact": None,
                "vqe": {"repetitions": [], "num_queries": None, "circuit_depth": None},
                "iqpe": {"repetitions": [], "iteration_energies": [], "num_queries": None, "circuit_depth": None},
            }
            for i in range(n_occ_count)
        },
    }

    def init_worker_logging():
        setup_logging()

    def append_progress(tag, result):
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

    def apply_result(tag, result):
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

    def load_progress():
        if not os.path.exists(progress_path):
            raise FileNotFoundError(f"Progress file does not exist: {progress_path}")
        with open(progress_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                apply_result(tuple(record["tag"]), record["result"])

    def validate_complete():
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

    def jobs_per_shard():
        value = os.environ.get("QBP_JOBS_PER_SHARD")
        if value:
            try:
                return max(1, int(value))
            except ValueError:
                pass
        try:
            return max(1, len(os.sched_getaffinity(0)))
        except AttributeError:
            return max(1, os.cpu_count() or 1)

    def write_outputs():
        validate_complete()

        with open(raw_data_path, "w") as f:
            json.dump(raw_data, f, indent=4)

        logger = setup_logging()
        exact = [raw_data["occupations"][str(i)]["exact"] for i in range(n_occ_count)]
        iqpe_data = [raw_data["occupations"][str(i)]["iqpe"]["repetitions"] for i in range(n_occ_count)]
        iqpe_best = [min(reps) for reps in iqpe_data]
        vqe_best = [min(raw_data["occupations"][str(i)]["vqe"]["repetitions"]) for i in range(n_occ_count)]

        for i in range(n_occ_count):
            logger.info(f"IQPE (n_sites={n_sites}, n_occ={i}) = {iqpe_best[i]}")
            logger.info(f"VQE (n_sites={n_sites}, n_occ={i}) = {vqe_best[i]}")

        data = {
            "result": {
                "exact": {i: exact[i] for i in range(n_occ_count)},
                "iqpe": {i: iqpe_best[i] for i in range(n_occ_count)},
                "vqe": {i: vqe_best[i] for i in range(n_occ_count)},
            },
            "num_queries": {
                "iqpe": {i: raw_data["occupations"][str(i)]["iqpe"]["num_queries"] for i in range(n_occ_count)},
                "vqe": {i: raw_data["occupations"][str(i)]["vqe"]["num_queries"] for i in range(n_occ_count)},
            },
            "circuit_depth": {
                "total": {
                    "iqpe": {i: raw_data["occupations"][str(i)]["iqpe"]["circuit_depth"]["total"] for i in range(n_occ_count)},
                    "vqe": {i: raw_data["occupations"][str(i)]["vqe"]["circuit_depth"]["total"] for i in range(n_occ_count)},
                },
                "two_qubit": {
                    "iqpe": {i: raw_data["occupations"][str(i)]["iqpe"]["circuit_depth"]["two_qubit"] for i in range(n_occ_count)},
                    "vqe": {i: raw_data["occupations"][str(i)]["vqe"]["circuit_depth"]["two_qubit"] for i in range(n_occ_count)},
                },
            },
        }

        log_path = os.path.join(REPO_ROOT, f"logs/{model.NAME}/{n_sites}-sites/simulated-{simulation_tag}-{suffix}.json")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w") as f:
            json.dump(data, f, indent=4)

        sim_label = "Ideal" if simulation_tag == "ideal" else "Noisy"
        param_str = ", ".join(f"${label}={model_params[k]}$" for k, label in model.PARAM_LABELS.items() if k in model_params)
        title = f"Real Space {model.DISPLAY_NAME} Hamiltonian Ground State Energy (Qiskit Aer {sim_label})\n{param_str}, $N_{{\\text{{sites}}}}={n_sites}$"

        plt.figure()
        plt.plot(range(n_occ_count), data["result"]["exact"].values(), "ro-", label="Exact")
        plt.plot(range(n_occ_count), data["result"]["iqpe"].values(), "go", label=f"IQPE (t={time_param}, n_trot={iqpe_trot}, n_iters={iqpe_iters}, n_reps={iqpe_reps})")
        plt.plot(range(n_occ_count), data["result"]["vqe"].values(), "bo", label=f"VQE (n_iters={vqe_iters}, n_layers={vqe_layers}, n_reps={vqe_reps})")
        plt.legend()
        plt.xlabel("Particle Number")
        plt.ylabel("$E$")
        plt.title(title, fontsize=11)
        plt.grid(True)
        plt.tight_layout()

        plot_path = os.path.join(REPO_ROOT, f"plots/{model.NAME}/{n_sites}-sites/simulated-{simulation_tag}-{suffix}.png")
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path)

    if args.task_count < 1:
        raise ValueError("--task-count must be at least 1")
    if args.task_index is not None and not 0 <= args.task_index < args.task_count:
        raise ValueError("--task-index must satisfy 0 <= task-index < task-count")

    if args.prepare_only:
        if not args.no_progress_log:
            with open(progress_path, "w") as f:
                f.write("")
        with open(raw_data_path, "w") as f:
            json.dump(raw_data, f, indent=4)
        raise SystemExit(0)

    if args.aggregate_only:
        load_progress()
        write_outputs()
        raise SystemExit(0)

    if args.task_index is None:
        if not args.no_progress_log:
            with open(progress_path, "w") as f:
                f.write("")
        job_results = Parallel(n_jobs=-1, return_as="generator_unordered", initializer=init_worker_logging)(jobs)
    else:
        init_worker_logging()
        shard_jobs = [job for idx, job in enumerate(jobs) if idx % args.task_count == args.task_index]
        job_results = Parallel(
            n_jobs=jobs_per_shard(),
            return_as="generator_unordered",
            initializer=init_worker_logging,
        )(shard_jobs)

    for tag, result in job_results:
        append_progress(tag, result)
        apply_result(tag, result)

    if args.task_index is not None:
        raise SystemExit(0)

    write_outputs()
