import math
import os

import qbp
from qbp import Method


MODEL = "haldane-honeycomb"
LATTICE = (3, 3)

_HERE = os.path.dirname(os.path.abspath(__file__))


qbp.run(
    model=MODEL,
    method=Method.ANALYTIC,
    lattice=LATTICE,
    boundary="open",
    x_param="n_occ",
    x_range=(0, 6, 1),
    model_params={"t1": 1.0, "t2": 0.1, "phi": math.pi / 4, "M": 0.0},
    log_path=os.path.join(_HERE, "logs", MODEL, "3x3", "hard-wall-n_occ.json"),
    plot_path=os.path.join(_HERE, "plots", MODEL, "3x3", "hard-wall-n_occ.pdf"),
    hide_plot=True,
)
