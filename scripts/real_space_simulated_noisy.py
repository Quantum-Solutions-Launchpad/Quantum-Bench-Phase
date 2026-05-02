import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
import os
import json
import argparse
import subprocess
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
from joblib import Parallel, delayed

from core import setup_logging, real_space_exact, real_space_vqe, real_space_iqpe, vqe_other_benchmarks, iqpe_other_benchmarks, resolve_sweep
from models import get_model

_N_OCC_DEFAULT = {"param": "n_occ", "range": None}

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
parser.add_argument("--n-occ", type=int, default=None,
                    help="Fixed particle number when neither sweep axis is n_occ (default: n_sites)")
parser.add_argument("--show-model-params", action="store_true", default=False,
                    help="Show bottom legend with fixed model parameters")
parser.add_argument("--show-sim-params", action="store_true", default=False,
                    help="Show top legend with VQE/IQPE simulation parameters")
parser.add_argument("--replot", action="store_true", default=False,
                    help="Skip computation and regenerate plot from existing log JSON")
args, _ = parser.parse_known_args()

model = get_model(args.model)
for param_name, default_val in model.DEFAULT_PARAMS.items():
    parser.add_argument(f"--{param_name}", type=type(default_val), default=default_val)

sweep_defs = getattr(model, "SWEEP_DEFAULTS", {})
x_def = sweep_defs.get("x", _N_OCC_DEFAULT)
y_def = sweep_defs.get("y", _N_OCC_DEFAULT)

parser.add_argument("--x-param", type=str, default=x_def["param"])
parser.add_argument("--x-range", type=float, nargs=3, metavar=("MIN", "MAX", "STEP"),
                    default=x_def["range"])
parser.add_argument("--y-param", type=str, default=y_def["param"])
parser.add_argument("--y-range", type=float, nargs=3, metavar=("MIN", "MAX", "STEP"),
                    default=y_def["range"])

args = parser.parse_args()
model_params = {k: getattr(args, k) for k in model.DEFAULT_PARAMS}

n_sites = args.n_sites
spin = 2
mapper = JordanWignerMapper()
vqe_iters, vqe_layers, vqe_reps = args.vqe_iters, args.vqe_layers, args.vqe_reps
time_param, iqpe_trot, iqpe_iters, iqpe_reps = args.iqpe_time, args.iqpe_trot, args.iqpe_iters, args.iqpe_reps
fixed_n_occ = args.n_occ if args.n_occ is not None else n_sites

x_vals, x_label, x_is_nocc = resolve_sweep(args.x_param, args.x_range, n_sites, spin)
y_vals, y_label, y_is_nocc = resolve_sweep(args.y_param, args.y_range, n_sites, spin)
if not x_is_nocc:
    x_label = f"${model.PARAM_LABELS.get(args.x_param, args.x_param)}$"
if not y_is_nocc:
    y_label = f"${model.PARAM_LABELS.get(args.y_param, args.y_param)}$"

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if args.replot:
    log_path = os.path.join(
        project_root,
        f"logs/{model.NAME}/{n_sites}-sites/simulated-noisy-{args.x_param}-vs-{args.y_param}.json"
    )
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"--replot requires an existing log at {log_path}; run without --replot first")
    with open(log_path) as f:
        data = json.load(f)
    x_vals = data["x_values"]
    y_vals = data["y_values"]
    nx, ny = len(x_vals), len(y_vals)
    Z_exact = np.array([[data["result"]["exact"][str(ix)][str(iy)] for iy in range(ny)] for ix in range(nx)])
    Z_vqe   = np.array([[data["result"]["vqe"][str(ix)][str(iy)]   for iy in range(ny)] for ix in range(nx)])
    Z_iqpe  = np.array([[data["result"]["iqpe"][str(ix)][str(iy)]  for iy in range(ny)] for ix in range(nx)])
else:
    def cell_params_and_nocc(ix, iy):
        params = model_params.copy()
        n_occ_val = fixed_n_occ
        xv, yv = x_vals[ix], y_vals[iy]
        if x_is_nocc:
            n_occ_val = int(xv)
        else:
            params[args.x_param] = xv
        if y_is_nocc:
            n_occ_val = int(yv)
        else:
            params[args.y_param] = yv
        return params, n_occ_val

    def tagged_job(tag, func, *a, **kw):
        return tag, func(*a, **kw)

    jobs = []
    for ix in range(len(x_vals)):
        for iy in range(len(y_vals)):
            cp, n_occ_val = cell_params_and_nocc(ix, iy)
            jobs.append(delayed(tagged_job)(
                ("exact", ix, iy), real_space_exact, model, n_sites, n_occ_val, cp
            ))
            for rep in range(1, iqpe_reps + 1):
                jobs.append(delayed(tagged_job)(
                    ("iqpe", ix, iy, rep), real_space_iqpe,
                    n_sites, n_occ_val, cp, model.fermionic_hamiltonian,
                    mapper, time_param, iqpe_trot, iqpe_iters, rep,
                    backend=backend
                ))
            for rep in range(1, vqe_reps + 1):
                jobs.append(delayed(tagged_job)(
                    ("vqe", ix, iy, rep), real_space_vqe,
                    n_sites, n_occ_val, cp, model.fermionic_hamiltonian, model.get_optimizer,
                    mapper, vqe_iters, vqe_layers, rep,
                    backend=backend
                ))
            jobs.append(delayed(tagged_job)(
                ("iqpe_bench", ix, iy), iqpe_other_benchmarks,
                n_sites, n_occ_val, cp, model.fermionic_hamiltonian,
                mapper, time_param, iqpe_trot, iqpe_iters, iqpe_reps,
                backend=backend
            ))
            jobs.append(delayed(tagged_job)(
                ("vqe_bench", ix, iy), vqe_other_benchmarks,
                n_sites, n_occ_val, cp, model.fermionic_hamiltonian,
                mapper, vqe_iters, vqe_layers, vqe_reps,
                backend=backend
            ))

    raw_data_path = os.path.join(
        project_root,
        f"logs/{model.NAME}/{n_sites}-sites/raw-data/simulated-noisy-{args.x_param}-vs-{args.y_param}.json"
    )
    os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)

    empty_cell = lambda: {
        "exact": None,
        "vqe": {"repetitions": [], "num_queries": None, "circuit_depth": None},
        "iqpe": {"repetitions": [], "iteration_energies": [], "num_queries": None, "circuit_depth": None},
    }

    raw_data = {
        "parameters": {
            "model": model.NAME,
            "n_sites": n_sites,
            "simulation": "noisy",
            "model_params": {k: float(v) for k, v in model_params.items()},
            "vqe": {"iters": vqe_iters, "layers": vqe_layers, "reps": vqe_reps},
            "iqpe": {"time": time_param, "trot": iqpe_trot, "iters": iqpe_iters, "reps": iqpe_reps},
        },
        "x_param": args.x_param, "y_param": args.y_param,
        "x_values": x_vals, "y_values": y_vals,
        "grid": {
            str(ix): {str(iy): empty_cell() for iy in range(len(y_vals))}
            for ix in range(len(x_vals))
        },
    }

    with open(raw_data_path, "w") as f:
        json.dump(raw_data, f, indent=4)

    def init_worker_logging():
        from core import setup_logging
        setup_logging(debug_enabled=not args.no_debug)

    for tag, result in Parallel(n_jobs=-1, return_as="generator_unordered", initializer=init_worker_logging)(jobs):
        ix, iy = str(tag[1]), str(tag[2])
        cell = raw_data["grid"][ix][iy]
        if tag[0] == "exact":
            cell["exact"] = result
        elif tag[0] == "iqpe":
            energy, iter_energies = result
            cell["iqpe"]["repetitions"].append(energy)
            cell["iqpe"]["iteration_energies"].append(iter_energies)
        elif tag[0] == "vqe":
            cell["vqe"]["repetitions"].append(result)
        elif tag[0] == "iqpe_bench":
            num_q, (total, two_q) = result
            cell["iqpe"]["num_queries"] = num_q
            cell["iqpe"]["circuit_depth"] = {"total": total, "two_qubit": two_q}
        elif tag[0] == "vqe_bench":
            num_q, (total, two_q) = result
            cell["vqe"]["num_queries"] = num_q
            cell["vqe"]["circuit_depth"] = {"total": total, "two_qubit": two_q}
        with open(raw_data_path, "w") as f:
            json.dump(raw_data, f, indent=4)

    logger = setup_logging(debug_enabled=not args.no_debug)

    nx, ny = len(x_vals), len(y_vals)
    Z_exact = np.full((nx, ny), np.nan)
    Z_vqe   = np.full((nx, ny), np.nan)
    Z_iqpe  = np.full((nx, ny), np.nan)

    for ix in range(nx):
        for iy in range(ny):
            cell = raw_data["grid"][str(ix)][str(iy)]
            Z_exact[ix, iy] = cell["exact"]
            Z_vqe[ix, iy]   = min(cell["vqe"]["repetitions"])
            Z_iqpe[ix, iy]  = min(cell["iqpe"]["repetitions"])
            logger.info(f"IQPE ({args.x_param}={x_vals[ix]}, {args.y_param}={y_vals[iy]}) = {Z_iqpe[ix, iy]}")
            logger.info(f"VQE  ({args.x_param}={x_vals[ix]}, {args.y_param}={y_vals[iy]}) = {Z_vqe[ix, iy]}")

    data = {
        "x_param": args.x_param, "y_param": args.y_param,
        "x_values": x_vals, "y_values": y_vals,
        "result": {
            "exact": {ix: {iy: Z_exact[ix, iy] for iy in range(ny)} for ix in range(nx)},
            "iqpe":  {ix: {iy: Z_iqpe[ix, iy]  for iy in range(ny)} for ix in range(nx)},
            "vqe":   {ix: {iy: Z_vqe[ix, iy]   for iy in range(ny)} for ix in range(nx)},
        },
        "num_queries": {
            "iqpe": {ix: {iy: raw_data["grid"][str(ix)][str(iy)]["iqpe"]["num_queries"] for iy in range(ny)} for ix in range(nx)},
            "vqe":  {ix: {iy: raw_data["grid"][str(ix)][str(iy)]["vqe"]["num_queries"]  for iy in range(ny)} for ix in range(nx)},
        },
        "circuit_depth": {
            "total": {
                "iqpe": {ix: {iy: raw_data["grid"][str(ix)][str(iy)]["iqpe"]["circuit_depth"]["total"] for iy in range(ny)} for ix in range(nx)},
                "vqe":  {ix: {iy: raw_data["grid"][str(ix)][str(iy)]["vqe"]["circuit_depth"]["total"]  for iy in range(ny)} for ix in range(nx)},
            },
            "two_qubit": {
                "iqpe": {ix: {iy: raw_data["grid"][str(ix)][str(iy)]["iqpe"]["circuit_depth"]["two_qubit"] for iy in range(ny)} for ix in range(nx)},
                "vqe":  {ix: {iy: raw_data["grid"][str(ix)][str(iy)]["vqe"]["circuit_depth"]["two_qubit"]  for iy in range(ny)} for ix in range(nx)},
            },
        },
    }

    log_path = os.path.join(
        project_root,
        f"logs/{model.NAME}/{n_sites}-sites/simulated-noisy-{args.x_param}-vs-{args.y_param}.json"
    )
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(data, f, indent=4)

def fmt_param(v):
    return round(v, 3) if isinstance(v, float) else v

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 13,
    "figure.dpi": 150,
})

fig = plt.figure(figsize=(10, 7.5))
ax = fig.add_subplot(111, projection="3d")

for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
    axis.pane.fill = False
    axis.pane.set_edgecolor("#cccccc")
ax.grid(True, linestyle="--", alpha=0.4)
ax.view_init(elev=25, azim=-55)
ax.dist = 7

cmap_obj = LinearSegmentedColormap.from_list("magma_dark", plt.cm.magma(np.linspace(0.05, 0.82, 256)))
ny = len(y_vals)
X_grid, Y_grid = np.meshgrid(x_vals, y_vals, indexing="ij")

ax.plot_surface(X_grid, Y_grid, Z_exact, cmap=cmap_obj, alpha=0.10, edgecolor="none",
                rcount=ny, ccount=len(x_vals))
for iy, yv in enumerate(y_vals):
    color = cmap_obj(iy / max(ny - 1, 1))
    ax.plot(x_vals, [yv] * len(x_vals), Z_exact[:, iy], color=color, linewidth=1.8, alpha=0.9, zorder=4)

ax.scatter(X_grid.ravel(), Y_grid.ravel(), Z_vqe.ravel(),
           color="#0072B2", marker="o", s=45, depthshade=True, zorder=6)
ax.scatter(X_grid.ravel(), Y_grid.ravel(), Z_iqpe.ravel(),
           color="#CC79A7", marker="^", s=45, depthshade=True, zorder=6)

ax.set_xlabel(x_label, labelpad=12)
ax.set_ylabel(y_label, labelpad=12)
ax.set_zlabel("$E$", labelpad=10)

if args.show_sim_params:
    vqe_label_str  = f"VQE (n_iters={vqe_iters}, n_layers={vqe_layers}, n_reps={vqe_reps})"
    iqpe_label_str = f"IQPE (t={time_param}, n_trot={iqpe_trot}, n_iters={iqpe_iters}, n_reps={iqpe_reps})"
    exact_proxy = mpatches.Patch(color=cmap_obj(0.7), alpha=0.9, label="Exact")
    vqe_proxy   = Line2D([0], [0], marker="o", color="w", markerfacecolor="#0072B2", markersize=10, label=vqe_label_str)
    iqpe_proxy  = Line2D([0], [0], marker="^", color="w", markerfacecolor="#CC79A7", markersize=10, label=iqpe_label_str)
    ax.legend(handles=[exact_proxy, vqe_proxy, iqpe_proxy], loc="upper center",
              ncol=3, fontsize=9, bbox_to_anchor=(0.5, 1.0))

if args.show_model_params:
    param_labels = [f"${label}={fmt_param(model_params[k])}$" for k, label in model.PARAM_LABELS.items() if k in model_params]
    param_labels.append(f"$N_{{\\text{{sites}}}}={n_sites}$")
    param_handles = [mpatches.Patch(fill=False, edgecolor="none", linewidth=0) for _ in param_labels]
    fig.legend(handles=param_handles, labels=param_labels, loc="lower center",
               ncol=len(param_labels), fontsize=14,
               handlelength=0, handletextpad=0,
               frameon=True, bbox_to_anchor=(0.5, 0.05))
    plt.tight_layout(rect=[0, 0.1, 1, 1.0])
else:
    plt.tight_layout()

plot_path = os.path.join(
    project_root,
    f"plots/{model.NAME}/{n_sites}-sites/simulated-noisy-{args.x_param}-vs-{args.y_param}.pdf"
)
os.makedirs(os.path.dirname(plot_path), exist_ok=True)
plt.savefig(plot_path, format="pdf")
subprocess.run(["pdfcrop", plot_path, plot_path], check=True, capture_output=True)
