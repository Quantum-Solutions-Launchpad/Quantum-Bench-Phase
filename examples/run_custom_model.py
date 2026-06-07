import os
import numpy as np
from qiskit_algorithms.optimizers import SPSA

import quaph
from quaph import Model, Method

# ---------------------------------------------------------------------------
# SSH (Su-Schrieffer-Heeger) model — 1D chain with staggered hopping.
#
# Alternating intracell (t1) and intercell (t2) hopping on a bipartite chain:
#   H = Σ_i  -t  (c†_i c_{i+1} + h.c.)   where t = t1 for even i, t2 for odd i
#
# The model undergoes a topological phase transition at |t2/t1| = 1:
#   trivial phase    |t2| < |t1|  (no in-gap edge states)
#   topological phase  |t2| > |t1|  (zero-energy edge states)
#
# This script demonstrates the in-process Python API path. The same model
# can also be registered interactively through the QuaPh console — see
# run_custom_model.sh.
# ---------------------------------------------------------------------------

def _ssh_optimizer(max_iters):
    return SPSA(maxiter=max_iters)


def _ssh_H_matrix(lattice, t1, t2):
    L, = lattice
    H = np.zeros((L, L), dtype=complex)
    for i in range(L - 1):
        t = t1 if i % 2 == 0 else t2
        H[i, i + 1] -= t
        H[i + 1, i] -= t
    return H


ssh_model = Model(
    name="ssh",
    display_name="SSH",
    param_labels={"t1": "t_1", "t2": "t_2"},
    spin=1,
    n_dims=1,
    lattice_shape=("L",),
    sites_per_cell=1,
    hamiltonian_matrix=_ssh_H_matrix,
    get_optimizer=_ssh_optimizer,
)

try:
    quaph.get_model("ssh")
except ValueError:
    quaph.register_model(ssh_model)

# ---------------------------------------------------------------------------

MODEL = "ssh"
LATTICE = (4,)
X_PARAM = "n_occ"
Y_PARAM = "t2"
METHODS = [Method.ANALYTIC, Method.VQE, Method.IQPE]

_HERE = os.path.dirname(os.path.abspath(__file__))
_LOG = os.path.join(_HERE, "logs", MODEL, "sim-n_occ-vs-t2.json")
_PLOT = os.path.join(_HERE, "plots", MODEL, "sim-n_occ-vs-t2.pdf")

if os.path.exists(_LOG):
    print("Plotting from existing log...")
    result = quaph.load_result(_LOG)
    result.plot()
else:
    result = quaph.run(
        model=MODEL,
        method=METHODS,
        lattice=LATTICE,
        x_param=X_PARAM,
        y_param=Y_PARAM,
        y_range=(0.0, 2.0, 0.5),
        model_params={"t1": 1.0},
        method_params={
            Method.VQE: {"iters": 200, "layers": 2, "reps": 1},
            Method.IQPE: {"time": 0.3, "trot": 2, "iters": 2, "reps": 1},
        },
        log_path=_LOG,
        plot_path=_PLOT,
    )
