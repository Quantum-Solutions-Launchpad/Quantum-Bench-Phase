import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
import os
import json
import argparse

from core import setup_logging, real_space_exact, resolve_sweep
from models import get_model

_N_OCC_DEFAULT = {"param": "n_occ", "range": None}

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--n-sites", type=int, default=6)
parser.add_argument("--n-occ", type=int, default=None,
                    help="Fixed particle number when neither sweep axis is n_occ (default: n_sites)")
parser.add_argument("--show-model-params", action="store_true", default=False,
                    help="Show bottom legend with fixed model parameters")
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

setup_logging()

n_sites = args.n_sites
spin = 2
fixed_n_occ = args.n_occ if args.n_occ is not None else n_sites

x_vals, x_label, x_is_nocc = resolve_sweep(args.x_param, args.x_range, n_sites, spin)
y_vals, y_label, y_is_nocc = resolve_sweep(args.y_param, args.y_range, n_sites, spin)
if not x_is_nocc:
    x_label = f"${model.PARAM_LABELS.get(args.x_param, args.x_param)}$"
if not y_is_nocc:
    y_label = f"${model.PARAM_LABELS.get(args.y_param, args.y_param)}$"

Z = np.full((len(x_vals), len(y_vals)), np.nan)

for ix, xv in enumerate(x_vals):
    for iy, yv in enumerate(y_vals):
        params = model_params.copy()
        n_occ_val = fixed_n_occ
        if x_is_nocc:
            n_occ_val = int(xv)
        else:
            params[args.x_param] = xv
        if y_is_nocc:
            n_occ_val = int(yv)
        else:
            params[args.y_param] = yv
        Z[ix, iy] = real_space_exact(model, n_sites, n_occ_val, params)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

log_path = os.path.join(
    project_root,
    f"logs/{model.NAME}/{n_sites}-sites/exact-{args.x_param}-vs-{args.y_param}.json"
)
os.makedirs(os.path.dirname(log_path), exist_ok=True)
with open(log_path, "w") as f:
    json.dump({
        "x_param": args.x_param, "y_param": args.y_param,
        "x_values": x_vals, "y_values": y_vals,
        "result": {"exact": {ix: {iy: Z[ix, iy] for iy in range(len(y_vals))}
                             for ix in range(len(x_vals))}},
    }, f, indent=4)

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

ax.plot_surface(X_grid, Y_grid, Z, cmap=cmap_obj, alpha=0.10, edgecolor="none",
                rcount=ny, ccount=len(x_vals))
for iy, yv in enumerate(y_vals):
    color = cmap_obj(iy / max(ny - 1, 1))
    ax.plot(x_vals, [yv] * len(x_vals), Z[:, iy], color=color, linewidth=1.8, alpha=0.95, zorder=5)

ax.set_xlabel(x_label, labelpad=12)
ax.set_ylabel(y_label, labelpad=12)
ax.set_zlabel("$E$", labelpad=10)

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

file_path = os.path.join(
    project_root,
    f"plots/{model.NAME}/{n_sites}-sites/exact-{args.x_param}-vs-{args.y_param}.pdf"
)
os.makedirs(os.path.dirname(file_path), exist_ok=True)
plt.savefig(file_path, format="pdf", bbox_inches="tight")
