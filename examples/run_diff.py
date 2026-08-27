import os
import qbp

MODEL   = "haldane-honeycomb"
LATTICE = (2, 2)
X_PARAM = "n_occ"
Y_PARAM = "t2"

_HERE        = os.path.dirname(os.path.abspath(__file__))
_LATTICE_TAG = "x".join(str(x) for x in LATTICE)
_LOG = os.path.join(
    _HERE, f"logs/{MODEL}/{_LATTICE_TAG}/simulated-ideal-3d-{X_PARAM}-vs-{Y_PARAM}.json"
)

# Diff plots between all method pairs are produced alongside the normal plot
# when diff=True. Reload a prior run and re-plot with diffs enabled:
qbp.load_result(_LOG).plot(
    diff=True,
    diff_format="3d",       # bar_2d | 3d | heatmap
    output_path=os.path.join(_HERE, "plots/diff.pdf"),
)