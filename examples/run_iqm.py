"""Run VQE/IQPE on an IQM Resonance device.

Requires the IQM client (``pip install qbp[iqm]``) and an IQM Resonance API token.
Generate the token on the IQM Resonance dashboard (Dashboard -> Generate token; it
is non-recoverable, so copy it immediately) and export it before running:

    export IQM_TOKEN=<your token>

Set BACKEND to one of the supported devices: "iqm_emerald", "iqm_garnet", or
"iqm_sirius".
"""

import os
import qbp
from qbp import Method

MODEL = "hubbard-honeycomb"
LATTICE = (1, 2)
X_PARAM = "n_occ"
Y_PARAM = "U"
METHODS = [Method.ANALYTIC, Method.VQE, Method.IQPE]

BACKEND = "iqm_garnet"

_HERE = os.path.dirname(os.path.abspath(__file__))
_LOG = os.path.join(_HERE, "logs", MODEL, "iqm-n_occ-vs-U.json")
_PLOT = os.path.join(_HERE, "plots", MODEL, "iqm-n_occ-vs-U.pdf")

if os.path.exists(_LOG):
    print("Plotting from existing log...")
    result = qbp.load_result(_LOG)
    result.plot()
else:
    result = qbp.run(
        model=MODEL,
        method=METHODS,
        backend=BACKEND,
        lattice=LATTICE,
        x_param=X_PARAM,
        y_param=Y_PARAM,
        y_range=(0.0, 4.0, 2.0),
        model_params={"t": 1.0},
        method_params={
            Method.VQE: {"iters": 20, "layers": 1, "reps": 1},
            Method.IQPE: {"time": 0.2, "trot": 1, "iters": 1, "reps": 1},
        },
        log_path=_LOG,
        plot_path=_PLOT,
    )
