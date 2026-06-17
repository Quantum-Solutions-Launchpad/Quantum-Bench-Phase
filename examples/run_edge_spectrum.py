import math
import os

import qbp


MODEL = "haldane"
LATTICE = (6, 6)

_HERE = os.path.dirname(os.path.abspath(__file__))
_PLOT_DIR = os.path.join(_HERE, "plots", MODEL, "6x6")


qbp.plot_edge_spectrum(
    model=MODEL,
    lattice=LATTICE,
    boundary="hard_wall",
    model_params={
        "t1": 1.0,
        "t2": 0.1,
        "phi": math.pi / 2,
        "M": 0.2,
    },
    output_path=os.path.join(_PLOT_DIR, "edge-spectrum-hard-wall.pdf"),
    hide_plot=True,
)
