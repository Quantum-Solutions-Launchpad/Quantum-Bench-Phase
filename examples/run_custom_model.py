import os
import numpy as np
from qiskit_algorithms.optimizers import SPSA

import quaph
from quaph import Model

# ---------------------------------------------------------------------------
# SSH (Su-Schrieffer-Heeger) model — 1D chain with staggered hopping.
#
# Alternating intracell (t1) and intercell (t2) hopping on a bipartite chain:
#   H = Σ_i  -t  (c†_i c_{i+1} + h.c.)   where t = t1 for even i, t2 for odd i
#
# The model undergoes a topological phase transition at |t2/t1| = 1:
#   trivial phase    |t2| < |t1|  (no in-gap edge states)
#   topological phase  |t2| > |t1|  (zero-energy edge states)
# ---------------------------------------------------------------------------

def _ssh_optimizer(max_iters):
    return SPSA(maxiter=max_iters)


def _ssh_H_matrix(n_sites, t1, t2):
    spin = 2
    H = np.zeros((n_sites * spin, n_sites * spin), dtype=complex)
    for i in range(n_sites - 1):
        t = t1 if i % 2 == 0 else t2
        for s in range(spin):
            s1 = i * spin + s
            s2 = (i + 1) * spin + s
            H[s1, s2] -= t
            H[s2, s1] -= t
    return H


ssh_model = Model(
    name="ssh",
    display_name="SSH",
    default_params={"t1": 1.0},
    param_labels={"t1": "t_1", "t2": "t_2"},
    hamiltonian_matrix=_ssh_H_matrix,
    get_optimizer=_ssh_optimizer,
    sweep_defaults={"y": {"param": "t2", "range": (0.0, 2.0, 0.25)}},
)

quaph.register_model(ssh_model)

# ---------------------------------------------------------------------------

MODEL = "ssh"
N_SITES = 4
X_PARAM = "n_occ"
Y_PARAM = "t2"

_HERE = os.path.dirname(os.path.abspath(__file__))
_LOG = os.path.join(_HERE, f"logs/{MODEL}/{N_SITES}-sites/simulated-ideal-{X_PARAM}-vs-{Y_PARAM}.json")

if os.path.exists(_LOG):
    print("Plotting from existing log...")
    result = quaph.load_result(_LOG)
    result.plot()
else:
    result = quaph.run_simulated_ideal(
        model=MODEL,
        n_sites=N_SITES,
        x_param=X_PARAM,
        y_param=Y_PARAM,
        vqe_iters=500,
        vqe_layers=2,
        vqe_reps=1,
        iqpe_time=0.3,
        iqpe_trot=2,
        iqpe_iters=4,
        iqpe_reps=1,
        log_dir=os.path.join(_HERE, "logs"),
        plot_dir=os.path.join(_HERE, "plots"),
    )
