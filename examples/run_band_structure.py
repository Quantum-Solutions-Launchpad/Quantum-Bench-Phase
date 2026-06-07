import os
import math
import quaph
from quaph import Method

MODEL = "haldane"
X_PARAM = "kx"
Y_PARAM = "ky"
STEP = math.pi / 50
MODEL_PARAMS = {"t1": 1.0, "t2": 0.05, "M": 0.2, "phi": math.pi / 2}

_HERE = os.path.dirname(os.path.abspath(__file__))

quaph.run(
    model=MODEL,
    method=[Method.ANALYTIC],
    x_param=X_PARAM,
    x_range=(-math.pi, math.pi, STEP),
    y_param=Y_PARAM,
    y_range=(-math.pi, math.pi, STEP),
    model_params=MODEL_PARAMS,
    log_path=os.path.join(_HERE, "logs", "haldane", "band-structure-heatmap.json"),
    plot_path=os.path.join(_HERE, "plots", "haldane", "band-structure-heatmap.pdf"),
    heatmap=True,
)

quaph.run(
    model=MODEL,
    method=[Method.ANALYTIC],
    x_param=X_PARAM,
    x_range=(-math.pi, math.pi, STEP),
    y_param=Y_PARAM,
    y_range=(-math.pi, math.pi, STEP),
    model_params=MODEL_PARAMS,
    log_path=os.path.join(_HERE, "logs", "haldane", "band-structure-3d.json"),
    plot_path=os.path.join(_HERE, "plots", "haldane", "band-structure-3d.pdf"),
    heatmap=False,
)
