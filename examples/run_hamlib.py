import os
import qbp
from qbp import Method

# Transverse-Field Ising Model (TFIM) from the Hamlib dataset.
#
# Plots the analytic ground-state energy sweeping both system size (Lx) and
# transverse field (h) for a 1D open-boundary chain.  The quantum phase
# transition at h = 1 is clearly visible in the 2D line plot.

HAMLIB_PATH = "/Users/adamgodel/hamlib/condensedmatter/tfim/tfim.zip"

_HERE = os.path.dirname(os.path.abspath(__file__))

# 3D surface: ground-state energy vs Lx and h
qbp.run(
    method=[Method.ANALYTIC],
    qubit_operator=HAMLIB_PATH,
    x_param="Lx",
    x_range=(4, 12),
    y_param="h",
    select=["1D", "nonpbc"],
    log_path=os.path.join(_HERE, "logs", "tfim", "Lx-vs-h.json"),
    plot_path=os.path.join(_HERE, "plots", "tfim", "Lx-vs-h.pdf"),
)

# Heatmap: same data, alternative view
qbp.run(
    method=[Method.ANALYTIC],
    qubit_operator=HAMLIB_PATH,
    x_param="Lx",
    x_range=(4, 12),
    y_param="h",
    select=["1D", "nonpbc"],
    heatmap=True,
    log_path=os.path.join(_HERE, "logs", "tfim", "Lx-vs-h-heatmap.json"),
    plot_path=os.path.join(_HERE, "plots", "tfim", "Lx-vs-h-heatmap.pdf"),
)

# 2D line: energy vs transverse field at fixed Lx=8
qbp.run(
    method=[Method.ANALYTIC],
    qubit_operator=HAMLIB_PATH,
    x_param="h",
    select=["1D", "nonpbc", "Lx-8"],
    log_path=os.path.join(_HERE, "logs", "tfim", "h.json"),
    plot_path=os.path.join(_HERE, "plots", "tfim", "h.pdf"),
)
