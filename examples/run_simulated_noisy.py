import os
import quaph

MODEL = "hubbard"
N_SITES = 8
X_PARAM = "n_occ"
Y_PARAM = "U"

_HERE = os.path.dirname(os.path.abspath(__file__))
_LOG = os.path.join(_HERE, f"logs/{MODEL}/{N_SITES}-sites/simulated-noisy-3d-{X_PARAM}-vs-{Y_PARAM}.json")

if os.path.exists(_LOG):
    print("Plotting from existing log...")
    result = quaph.load_result(_LOG)
    result.plot()
else:
    result = quaph.run_simulated_noisy(
        model=MODEL,
        n_sites=N_SITES,
        x_param=X_PARAM,
        y_param=Y_PARAM,
        y_range=(0.0, 4.0, 1.0),
        model_params={"t": 1.0},
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
