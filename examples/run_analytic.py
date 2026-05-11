import os
import quaph

MODEL = "haldane"
N_SITES = 6
X_PARAM = "n_occ"
Y_PARAM = "t2"

_HERE = os.path.dirname(os.path.abspath(__file__))

result = quaph.run_analytic(
    model=MODEL,
    n_sites=N_SITES,
    x_param=X_PARAM,
    y_param=Y_PARAM,
    log_dir=os.path.join(_HERE, "logs"),
    plot_dir=os.path.join(_HERE, "plots"),
)
