import math
import os

import qbp


MODEL = "haldane"
PARENT_LATTICE = (18, 18)

_HERE = os.path.dirname(os.path.abspath(__file__))
_PLOT_DIR = os.path.join(_HERE, "plots", MODEL, "profiles")


PARAMS = {
    "t1": 1.0,
    "t2": 0.1,
    "phi": math.pi / 2,
    "M": 0.2,
}


# Phase 2: scalar soft confinement wall inside a larger hard-wall flake.
qbp.plot_real_space_state_density(
    model=MODEL,
    lattice=PARENT_LATTICE,
    boundary="hard_wall",
    model_params=PARAMS,
    potential_profile="soft_dot",
    potential_radius=5.5,
    potential_v0=3.0,
    potential_xi=0.8,
    output_path=os.path.join(_PLOT_DIR, "soft-dot-density.pdf"),
    hide_plot=True,
)

qbp.plot_edge_spectrum(
    model=MODEL,
    lattice=PARENT_LATTICE,
    boundary="hard_wall",
    model_params=PARAMS,
    potential_profile="soft_dot",
    potential_radius=5.5,
    potential_v0=3.0,
    potential_xi=0.8,
    output_path=os.path.join(_PLOT_DIR, "soft-dot-edge-spectrum.pdf"),
    hide_plot=True,
)


# Phase 3: radial Semenoff-mass interface.
# M is set to 0 in the base Hamiltonian so the radial profile supplies M(r).
TOPOLOGICAL_INTERFACE_PARAMS = dict(PARAMS)
TOPOLOGICAL_INTERFACE_PARAMS["M"] = 0.0

qbp.plot_real_space_state_density(
    model=MODEL,
    lattice=PARENT_LATTICE,
    boundary="hard_wall",
    model_params=TOPOLOGICAL_INTERFACE_PARAMS,
    mass_profile="radial_tanh",
    mass_radius=5.5,
    mass_inner=0.2,
    mass_outer=0.8,
    mass_xi=0.8,
    output_path=os.path.join(_PLOT_DIR, "topological-interface-density.pdf"),
    hide_plot=True,
)

qbp.plot_edge_spectrum(
    model=MODEL,
    lattice=PARENT_LATTICE,
    boundary="hard_wall",
    model_params=TOPOLOGICAL_INTERFACE_PARAMS,
    mass_profile="radial_tanh",
    mass_radius=5.5,
    mass_inner=0.2,
    mass_outer=0.8,
    mass_xi=0.8,
    output_path=os.path.join(_PLOT_DIR, "topological-interface-edge-spectrum.pdf"),
    hide_plot=True,
)
