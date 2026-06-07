import os
import math
import quaph
from quaph import Method

# DMRG vs. exact ground-state energy across a Haldane-model phase diagram.
#
# DMRG requires a working Julia + ITensorMPS toolchain (see scripts/julia-dmrg/),
# so this script will only run end-to-end where Julia is available. The committed
# DMRG plot is retained as-is. Selecting Method.ANALYTIC alongside Method.DMRG
# overlays the exact ground-state energy as the reference surface.

MODEL = "haldane"
LATTICE = (2, 2)
X_PARAM = "n_occ"
Y_PARAM = "t2"
MODEL_PARAMS = {"t1": 1.0, "phi": math.pi / 4, "M": 0.0}

_HERE = os.path.dirname(os.path.abspath(__file__))
_LATTICE_TAG = "x".join(str(x) for x in LATTICE)
_LOG = os.path.join(
    _HERE,
    f"logs/{MODEL}/{_LATTICE_TAG}/run-analytic+dmrg-3d-{X_PARAM}-vs-{Y_PARAM}.json",
)

if os.path.exists(_LOG):
    print("Plotting from existing log: {}".format(_LOG))
    quaph.load_result(_LOG).plot()
else:
    result = quaph.run(
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
        log_dir=os.path.join(_HERE, "logs"),
        plot_dir=os.path.join(_HERE, "plots"),
    )
    print("Wrote {}".format(result.log_path))
