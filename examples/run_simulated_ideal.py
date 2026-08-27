import os
import math
import qbp
from qbp import Method

MODEL = "haldane-hubbard-honeycomb"
LATTICE = (2, 2)
X_PARAM = "t2"
Y_PARAM = "U"
METHODS = [Method.ANALYTIC, Method.VQE, Method.IQPE]

_HERE = os.path.dirname(os.path.abspath(__file__))
_LOG = os.path.join(_HERE, "logs", MODEL, "sim-ideal-t2-vs-U.json")
_PLOT = os.path.join(_HERE, "plots", MODEL, "sim-ideal-t2-vs-U.pdf")

if os.path.exists(_LOG):
    print("Plotting from existing log...")
    result = qbp.load_result(_LOG)
    result.plot(diff=True, diff_format="3d")

else:
    result = qbp.run(
        model=MODEL,
        method=METHODS,
        lattice=LATTICE,
        x_param=X_PARAM,
        x_range=(0.0, 1.5, 0.3),
        y_param=Y_PARAM,
        y_range=(0.0, 4.0, 1.0),
        model_params={"t1": 1.0, "phi": math.pi / 4, "M": 0.0},
        method_params={
            Method.VQE: {"iters": 200, "layers": 2, "reps": 1},
            Method.IQPE: {"time": 0.2, "trot": 2, "iters": 2, "reps": 1},
        },
        log_path=_LOG,
        plot_path=_PLOT,
    )
