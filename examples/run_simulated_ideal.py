import os
import math
import quaph

MODEL = "haldane-hubbard"
N_SITES = 8
X_PARAM = "t2"
Y_PARAM = "U"

_HERE = os.path.dirname(os.path.abspath(__file__))
_LOG = os.path.join(_HERE, f"logs/{MODEL}/{N_SITES}-sites/simulated-ideal-3d-{X_PARAM}-vs-{Y_PARAM}.json")

if os.path.exists(_LOG):
    print("Plotting from existing log...")
    result = quaph.load_result(_LOG)
    result.plot()
else:
    result = quaph.run_simulated_ideal(
        model=MODEL,
        n_sites=N_SITES,
        x_param=X_PARAM,
        x_range=(0.0, 1.5, 0.3),
        y_param=Y_PARAM,
        y_range=(0.0, 4.0, 1.0),
        model_params={"t1": 1.0, "phi": math.pi / 4, "M": 0.0},
        vqe_iters=200,
        vqe_layers=2,
        vqe_reps=1,
        iqpe_time=0.2,
        iqpe_trot=2,
        iqpe_iters=2,
        iqpe_reps=1,
        log_dir=os.path.join(_HERE, "logs"),
        plot_dir=os.path.join(_HERE, "plots"),
    )
