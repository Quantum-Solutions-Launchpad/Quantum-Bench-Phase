import os
import quaph

MODEL = "haldane-hubbard"
N_SITES = 6
X_PARAM = "t2"
Y_PARAM = "U"

_LOG = f"logs/{MODEL}/{N_SITES}-sites/simulated-ideal-{X_PARAM}-vs-{Y_PARAM}.json"

if os.path.exists(_LOG):
    result = quaph.load_result(_LOG)
    result.plot()
else:
    result = quaph.run_simulated_ideal(
        model=MODEL,
        n_sites=N_SITES,
        x_param=X_PARAM,
        y_param=Y_PARAM,
        vqe_iters=10000,
        vqe_layers=5,
        vqe_reps=10,
        iqpe_time=0.2,
        iqpe_trot=5,
        iqpe_iters=8,
        iqpe_reps=20,
        log_dir="logs",
        plot_dir="plots",
    )
