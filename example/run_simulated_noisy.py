import os
import quaph

MODEL = "hubbard"
N_SITES = 6
X_PARAM = "n_occ"
Y_PARAM = "U"

_LOG = f"logs/{MODEL}/{N_SITES}-sites/simulated-noisy-{X_PARAM}-vs-{Y_PARAM}.json"

if os.path.exists(_LOG):
    result = quaph.load_result(_LOG)
    result.plot()
else:
    result = quaph.run_simulated_noisy(
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
