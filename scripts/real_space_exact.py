import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

from core import setup_logging
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
    data[n_occ] = model.real_space_exact(n_sites, n_occ, **model_params)

def fmt_param(k, v):
    return round(v, 3) if isinstance(v, float) else v

param_str = ", ".join(f"${label}={fmt_param(k, model_params[k])}$" for k, label in model.PARAM_LABELS.items())
title = f"Real Space {model.DISPLAY_NAME} Hamiltonian Ground State Energy (Exact)\n{param_str}, $N_{{\\text{{sites}}}}={n_sites}$"

plt.figure()
plt.plot(range(spin * n_sites + 1), data.values(), 'ro-')
plt.xlabel("Particle Number")
plt.ylabel("$E$")
plt.title(title)
plt.grid(True, alpha=0.3)
plt.tight_layout()

suffix = model.file_suffix(model_params)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
file_path = os.path.join(project_root, f"plots/{model.NAME}/{n_sites}-sites/exact-{suffix}.png")
os.makedirs(os.path.dirname(file_path), exist_ok=True)
plt.savefig(file_path)
