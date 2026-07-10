import os
import math
import qbp
from qbp import Method

# DMRG vs. exact ground-state energy across a Haldane-model phase diagram.
#
# DMRG requires a working Julia + ITensorMPS toolchain (bundled under
# qbp/julia-dmrg/), so this script only runs end-to-end where Julia is
# available. Selecting Method.ANALYTIC alongside Method.DMRG overlays the exact
# ground-state energy as the reference surface.

MODEL = "haldane"
LATTICE = (2, 2)
X_PARAM = "n_occ"
Y_PARAM = "t2"
MODEL_PARAMS = {"t1": 1.0, "phi": math.pi / 4, "M": 0.0}

_HERE = os.path.dirname(os.path.abspath(__file__))
_LOG = os.path.join(_HERE, "logs", MODEL, "dmrg-n_occ-vs-t2.json")
_PLOT = os.path.join(_HERE, "plots", MODEL, "dmrg-n_occ-vs-t2.pdf")

if os.path.exists(_LOG):
    print("Plotting from existing log: {}".format(_LOG))
    qbp.load_result(_LOG).plot()
else:
    result = qbp.run(
        model=MODEL,
        method=[Method.ANALYTIC, Method.DMRG],
        lattice=LATTICE,
        x_param=X_PARAM,
        y_param=Y_PARAM,
        y_range=(0.0, 1.0, 0.1),
        model_params=MODEL_PARAMS,
        method_params={
            Method.DMRG: {
                "nsweeps": 4,
                "maxdims": "20,50,100,200",
                "cutoff": 1e-9,
            },
        },
        log_path=_LOG,
        plot_path=_PLOT,
    )
    print("Wrote {}".format(result.log_path))
