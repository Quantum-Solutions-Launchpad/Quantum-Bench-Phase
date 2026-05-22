import os
import math
import quaph

MODEL = "haldane"
X_PARAM = "kx"
Y_PARAM = "ky"
STEP = math.pi / 50
MODEL_PARAMS = {"t1": 1.0, "t2": 0.05, "M": 0.2, "phi": math.pi / 2}

_HERE = os.path.dirname(os.path.abspath(__file__))

quaph.run_analytic(
    model=MODEL,
    x_param=X_PARAM,
    x_range=(-math.pi, math.pi, STEP),
    y_param=Y_PARAM,
    y_range=(-math.pi, math.pi, STEP),
    model_params=MODEL_PARAMS,
    log_dir=os.path.join(_HERE, "logs"),
    plot_dir=os.path.join(_HERE, "plots"),
    heatmap=True,
)

quaph.run_analytic(
    model=MODEL,
    x_param=X_PARAM,
    x_range=(-math.pi, math.pi, STEP),
    y_param=Y_PARAM,
    y_range=(-math.pi, math.pi, STEP),
    model_params=MODEL_PARAMS,
    log_dir=os.path.join(_HERE, "logs"),
    plot_dir=os.path.join(_HERE, "plots"),
    heatmap=False,
)
