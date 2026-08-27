import math
import os

import qbp
from qbp import Method, SemenoffMass


MODEL = "haldane-honeycomb"
PARENT_LATTICE = (18, 18)

_HERE = os.path.dirname(os.path.abspath(__file__))
_PLOT_DIR = os.path.join(_HERE, "plots", MODEL, "18x18")


PARAMS = {
    "t1": 1.0,
    "t2": 0.1,
    "phi": math.pi / 2,
    "M": 0.2,
}


# Phase 2: scalar soft confinement wall inside a larger hard-wall flake.
qbp.run(
    model=MODEL,
    method=Method.ANALYTIC,
    lattice=PARENT_LATTICE,
    x_param="Lx",
    y_param="Ly",
    boundary="open",
    boundary_params={
        "potential_profile": "soft_dot",
        "potential_radius": 5.5,
        "potential_v0": 3.0,
        "potential_xi": 0.8,
    },
    model_params=PARAMS,
    plot_path=os.path.join(_PLOT_DIR, "soft-dot-density.pdf"),
    hide_plot=True,
)

qbp.run(
    model=MODEL,
    method=Method.ANALYTIC,
    lattice=PARENT_LATTICE,
    x_param="eigenstate",
    boundary="open",
    boundary_params={
        "potential_profile": "soft_dot",
        "potential_radius": 5.5,
        "potential_v0": 3.0,
        "potential_xi": 0.8,
    },
    model_params=PARAMS,
    plot_path=os.path.join(_PLOT_DIR, "soft-dot-edge-spectrum.pdf"),
    hide_plot=True,
)


# Phase 3: radial Semenoff-mass interface.
# M is set to 0 in the base Hamiltonian so the radial profile supplies M(r).
TOPOLOGICAL_INTERFACE_PARAMS = dict(PARAMS)
TOPOLOGICAL_INTERFACE_PARAMS["M"] = 0.0

qbp.run(
    model=MODEL,
    method=Method.ANALYTIC,
    lattice=PARENT_LATTICE,
    x_param="Lx",
    y_param="Ly",
    boundary="open",
    model_params=TOPOLOGICAL_INTERFACE_PARAMS,
    investigation=SemenoffMass(profile="radial_tanh", radius=5.5, inner=0.2, outer=0.8, xi=0.8),
    plot_path=os.path.join(_PLOT_DIR, "topological-interface-density.pdf"),
    hide_plot=True,
)

qbp.run(
    model=MODEL,
    method=Method.ANALYTIC,
    lattice=PARENT_LATTICE,
    x_param="eigenstate",
    boundary="open",
    model_params=TOPOLOGICAL_INTERFACE_PARAMS,
    investigation=SemenoffMass(profile="radial_tanh", radius=5.5, inner=0.2, outer=0.8, xi=0.8),
    plot_path=os.path.join(_PLOT_DIR, "topological-interface-edge-spectrum.pdf"),
    hide_plot=True,
)
