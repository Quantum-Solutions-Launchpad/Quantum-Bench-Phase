import math
import os

import quaph


MODEL = "haldane"
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


quaph.plot_real_space_state_density(
    model=MODEL,
    lattice=PARENT_LATTICE,
    boundary="hard_wall",
    geometry="disk",
    radius=RADIUS,
    model_params=PARAMS,
    output_path=os.path.join(_PLOT_DIR, "disk-density-hard-wall.pdf"),
    hide_plot=True,
)

quaph.plot_edge_spectrum(
    model=MODEL,
    lattice=PARENT_LATTICE,
    boundary="hard_wall",
    geometry="disk",
    radius=RADIUS,
    model_params=PARAMS,
    output_path=os.path.join(_PLOT_DIR, "disk-edge-spectrum-hard-wall.pdf"),
    hide_plot=True,
)
