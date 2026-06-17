import math
import os

import qbp
from qbp import Method


MODEL = "haldane"
LATTICE = (3, 3)

_HERE = os.path.dirname(os.path.abspath(__file__))
_PLOT_DIR = os.path.join(_HERE, "plots", MODEL, "3x3")


PARAMS = {
    "t1": 1.0,
    "t2": 0.1,
    "phi": math.pi / 4,
    "M": 0.0,
}


qbp.run(
    model=MODEL,
    method=Method.ANALYTIC,
    lattice=LATTICE,
    x_param="Lx",
    y_param="Ly",
    model_params=PARAMS,
    boundary="periodic",
    plot_path=os.path.join(_PLOT_DIR, "real-space-density-periodic-2d.pdf"),
    hide_plot=True,
)

qbp.run(
    model=MODEL,
    method=Method.ANALYTIC,
    lattice=LATTICE,
    x_param="Lx",
    y_param="Ly",
    model_params=PARAMS,
    boundary="hard_wall",
    plot_path=os.path.join(_PLOT_DIR, "real-space-density-hard-wall-2d.pdf"),
    hide_plot=True,
)
