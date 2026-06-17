import math
import os

import quaph


MODEL = "haldane"
LATTICE = (3, 3)

_HERE = os.path.dirname(os.path.abspath(__file__))


quaph.run_analytic(
    model=MODEL,
    lattice=LATTICE,
    boundary="hard_wall",
    x_param="n_occ",
    x_range=(0, 6, 1),
    model_params={"t1": 1.0, "t2": 0.1, "phi": math.pi / 4, "M": 0.0},
    log_dir=os.path.join(_HERE, "logs"),
    plot_dir=os.path.join(_HERE, "plots"),
)
