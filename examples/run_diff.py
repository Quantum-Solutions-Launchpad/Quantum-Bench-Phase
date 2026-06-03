import os
import quaph

MODEL   = "haldane"
LATTICE = (2, 2)
X_PARAM = "n_occ"
Y_PARAM = "t2"

_HERE        = os.path.dirname(os.path.abspath(__file__))
_LATTICE_TAG = "x".join(str(x) for x in LATTICE)
_LOG = os.path.join(
    _HERE, f"logs/{MODEL}/{_LATTICE_TAG}/simulated-ideal-3d-{X_PARAM}-vs-{Y_PARAM}.json"
)

quaph.plot_diff(
    _LOG,
    method="both",          # vqe | iqpe | both
    plot_format="3d",       # bar_2d | 3d  | heatmap
    output_path=os.path.join(_HERE, "plots/diff.pdf"),
)