import numpy as np
import matplotlib.pyplot as plt
import os
import json
import argparse

from core import setup_logging, real_space_exact
from models import get_model

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--n-sites", type=int, default=6)
args, _ = parser.parse_known_args()

model = get_model(args.model)
for param_name, default_val in model.DEFAULT_PARAMS.items():
    parser.add_argument(f"--{param_name}", type=type(default_val), default=default_val)
args = parser.parse_args()
model_params = {k: getattr(args, k) for k in model.DEFAULT_PARAMS}

setup_logging()

n_sites = args.n_sites
spin = 2

data = {}
for n_occ in range(spin * n_sites + 1):
    data[n_occ] = real_space_exact(model, n_sites, n_occ, model_params)

suffix = model.file_suffix(model_params)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

log_path = os.path.join(project_root, f"logs/{model.NAME}/{n_sites}-sites/exact-{suffix}.json")
os.makedirs(os.path.dirname(log_path), exist_ok=True)
with open(log_path, "w") as f:
    json.dump({"result": {"exact": {i: data[i] for i in range(spin * n_sites + 1)}}}, f, indent=4)

def fmt_param(v):
    return round(v, 3) if isinstance(v, float) else v

param_str = ", ".join(f"${label}={fmt_param(model_params[k])}$" for k, label in model.PARAM_LABELS.items())
title = f"Real Space {model.DISPLAY_NAME} Hamiltonian Ground State Energy (Exact)\n{param_str}, $N_{{\\text{{sites}}}}={n_sites}$"

plt.figure()
plt.plot(range(spin * n_sites + 1), data.values(), 'ro-')
plt.xlabel("Particle Number")
plt.ylabel("$E$")
plt.title(title)
plt.grid(True, alpha=0.3)
plt.tight_layout()

file_path = os.path.join(project_root, f"plots/{model.NAME}/{n_sites}-sites/exact-{suffix}.png")
os.makedirs(os.path.dirname(file_path), exist_ok=True)
plt.savefig(file_path)
