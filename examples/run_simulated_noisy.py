import os
import qbp
from qbp import Method
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke

MODEL = "hubbard"
LATTICE = (2, 2)
X_PARAM = "n_occ"
Y_PARAM = "U"
METHODS = [Method.ANALYTIC, Method.VQE, Method.IQPE]

_HERE = os.path.dirname(os.path.abspath(__file__))
_LOG = os.path.join(_HERE, "logs", MODEL, "sim-noisy-n_occ-vs-U.json")
_PLOT = os.path.join(_HERE, "plots", MODEL, "sim-noisy-n_occ-vs-U.pdf")

if os.path.exists(_LOG):
    print("Plotting from existing log...")
    result = qbp.load_result(_LOG)
    result.plot()
else:
    result = qbp.run(
        model=MODEL,
        method=METHODS,
        backend=FakeSherbrooke(),
        lattice=LATTICE,
        x_param=X_PARAM,
        y_param=Y_PARAM,
        y_range=(0.0, 4.0, 1.0),
        model_params={"t": 1.0},
        method_params={
            Method.VQE: {"iters": 200, "layers": 2, "reps": 1},
            Method.IQPE: {"time": 0.2, "trot": 2, "iters": 2, "reps": 1},
        },
        log_path=_LOG,
        plot_path=_PLOT,
    )
