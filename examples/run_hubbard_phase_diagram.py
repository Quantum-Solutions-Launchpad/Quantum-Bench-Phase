import os
import quaph

MODEL = "hubbard"
LATTICE = (3, 3)

_HERE = os.path.dirname(os.path.abspath(__file__))

common = dict(
    model=MODEL,
    lattice=LATTICE,
    x_param="n_occ",
    y_param="U",
    y_range=(0.0, 10.0, 0.5),
    model_params={"t": 1.0},
    log_dir=os.path.join(_HERE, "logs"),
    plot_dir=os.path.join(_HERE, "plots"),
    heatmap=True,
    hide_plot=True,
)

for obs in ("E_uhf", "M_stag", "M_total", "gap_uhf"):
    quaph.run_analytic(observable=obs, **common)
