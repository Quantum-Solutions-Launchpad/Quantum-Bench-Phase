import os
import math
import quaph

MODEL = "haldane"
N_SITES = 18
X_PARAM = "n_occ"
Y_PARAM = "t2"

_HERE = os.path.dirname(os.path.abspath(__file__))

quaph.run_analytic(
    model=MODEL,
    n_sites=N_SITES,
    x_param=X_PARAM,
    y_param=Y_PARAM,
    y_range=(0.0, 1.0, 0.1),
    model_params={"t1": 1.0, "phi": math.pi / 4, "M": 0.0},
    log_dir=os.path.join(_HERE, "logs"),
    plot_dir=os.path.join(_HERE, "plots"),
)
