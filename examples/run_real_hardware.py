import os
import qbp
from qbp import Method

MODEL = "hubbard"
LATTICE = (1, 2)
X_PARAM = "n_occ"
Y_PARAM = "U"
METHODS = [Method.ANALYTIC, Method.VQE, Method.IQPE]

# Real IBM device. Requires configured Qiskit Runtime credentials, e.g.:
#   from qiskit_ibm_runtime import QiskitRuntimeService
#   QiskitRuntimeService.save_account(
#       channel="ibm_quantum_platform", token="<TOKEN>", instance="<CRN>")
# Use a device name (e.g. "ibm_brisbane") or "least_busy" to auto-pick a device.
BACKEND = "least_busy"

_HERE = os.path.dirname(os.path.abspath(__file__))
_LOG = os.path.join(_HERE, "logs", MODEL, "real-hardware-n_occ-vs-U.json")
_PLOT = os.path.join(_HERE, "plots", MODEL, "real-hardware-n_occ-vs-U.pdf")

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
