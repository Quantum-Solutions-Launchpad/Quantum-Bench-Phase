import os
import math
import quaph
from quaph import Method
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke

MODEL   = "haldane"
LATTICE = (2, 2)
X_PARAM = "n_occ"
Y_PARAM = "t2"

_HERE        = os.path.dirname(os.path.abspath(__file__))
_LATTICE_TAG = "x".join(str(x) for x in LATTICE)

result = quaph.run(
    MODEL,
    method=[Method.ANALYTIC, Method.IQPE],
    method_params={
        "iqpe": {
            "time": 0.5,
            "trot": 4,
            "iters": 8,
            "reps": 3,
            "mitigation": {"m3": True},
        }
    },
    lattice=LATTICE,
    x_param=X_PARAM,
    y_param=Y_PARAM,
    y_range=(0.0, 1.0, 0.25),
    model_params={"t1": 1.0, "phi": math.pi / 4, "M": 0.1},
    backend=FakeSherbrooke(),
    log_path=os.path.join(_HERE, f"logs/{MODEL}/{_LATTICE_TAG}/iqpe-m3-3d-{X_PARAM}-vs-{Y_PARAM}.json"),
    plot_path=os.path.join(_HERE, f"plots/{MODEL}/{_LATTICE_TAG}/iqpe-m3-3d-{X_PARAM}-vs-{Y_PARAM}.pdf"),
)