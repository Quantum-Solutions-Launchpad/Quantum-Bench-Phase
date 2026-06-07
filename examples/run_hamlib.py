import os
import quaph
from quaph import Method

# Transverse-Field Ising Model (TFIM) from the Hamlib dataset.
#
# Plots the analytic ground-state energy sweeping both system size (Lx) and
# transverse field (h) for a 1D open-boundary chain.  The quantum phase
# transition at h = 1 is clearly visible in the 2D line plot.

HAMLIB_PATH = "/Users/adamgodel/hamlib/condensedmatter/tfim/tfim.zip"

_HERE = os.path.dirname(os.path.abspath(__file__))

# 3D surface: ground-state energy vs Lx and h
quaph.run(
    method=[Method.ANALYTIC],
    qubit_operator=HAMLIB_PATH,
    x_param="Lx",
    x_range=(4, 12),
    y_param="h",
    select=["1D", "nonpbc"],
    log_dir=os.path.join(_HERE, "logs", "tfim"),
    plot_dir=os.path.join(_HERE, "plots", "tfim"),
)

# Heatmap: same data, alternative view
quaph.run(
    method=[Method.ANALYTIC],
    qubit_operator=HAMLIB_PATH,
    x_param="Lx",
    x_range=(4, 12),
    y_param="h",
    select=["1D", "nonpbc"],
    heatmap=True,
    log_dir=os.path.join(_HERE, "logs", "tfim"),
    plot_dir=os.path.join(_HERE, "plots", "tfim"),
)

# 2D line: energy vs transverse field at fixed Lx=8
quaph.run(
    method=[Method.ANALYTIC],
    qubit_operator=HAMLIB_PATH,
    x_param="h",
    select=["1D", "nonpbc", "Lx-8"],
    log_dir=os.path.join(_HERE, "logs", "tfim"),
    plot_dir=os.path.join(_HERE, "plots", "tfim"),
)
