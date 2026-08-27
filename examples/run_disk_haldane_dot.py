import math
import os

import qbp
from qbp import Method


MODEL = "haldane-honeycomb"
PARENT_LATTICE = (14, 14)
RADIUS = 5.5

_HERE = os.path.dirname(os.path.abspath(__file__))
_PLOT_DIR = os.path.join(_HERE, "plots", MODEL, "disk")


PARAMS = {
    "t1": 1.0,
    "t2": 0.1,
    "phi": math.pi / 2,
    "M": 0.2,
}


qbp.run(
    model=MODEL,
    method=Method.ANALYTIC,
    lattice=PARENT_LATTICE,
    x_param="Lx",
    y_param="Ly",
    boundary="open",
    boundary_params={"geometry": "disk", "radius": RADIUS},
    model_params=PARAMS,
    plot_path=os.path.join(_PLOT_DIR, "disk-density-hard-wall.pdf"),
    hide_plot=True,
)

qbp.run(
    model=MODEL,
    method=Method.ANALYTIC,
    lattice=PARENT_LATTICE,
    x_param="eigenstate",
    boundary="open",
    boundary_params={"geometry": "disk", "radius": RADIUS},
    model_params=PARAMS,
    plot_path=os.path.join(_PLOT_DIR, "disk-edge-spectrum-hard-wall.pdf"),
    hide_plot=True,
)
