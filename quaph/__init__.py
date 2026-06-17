from quaph._core import setup_logging as _setup_logging
_setup_logging()

from quaph._model import Model, ModelCapabilityError
from quaph._registry import (
    get_model,
    register_model,
    register_model_from_file,
    remove_model,
)
from quaph._yaml_model import build_tight_binding_model
from quaph._hamlib import load_hamlib_operator, list_hamlib_keys
from quaph._method import Method
from quaph._run import (
    run,
    load_result,
    RunResult,
)
from quaph._realspace import (
    plot_real_space_state_density,
    real_space_positions,
    RealSpaceStateResult,
)
from quaph._edge import (
    plot_edge_spectrum,
    edge_mask_from_missing_bonds,
    edge_participation_all,
    inverse_participation_ratio_all,
    EdgeSpectrumResult,
)
from quaph._geometry import (
    geometry_projection,
    apply_geometry_to_hamiltonian,
    GeometryProjection,
)
from quaph._profiles import (
    soft_dot_potential,
    radial_mass_values,
    apply_profiles_to_hamiltonian,
)

__all__ = [
    "Model",
    "ModelCapabilityError",
    "get_model",
    "register_model",
    "register_model_from_file",
    "build_tight_binding_model",
    "remove_model",
    "load_hamlib_operator",
    "list_hamlib_keys",
    "Method",
    "run",
    "load_result",
    "RunResult",
    "plot_real_space_state_density",
    "real_space_positions",
    "RealSpaceStateResult",
    "plot_edge_spectrum",
    "edge_mask_from_missing_bonds",
    "edge_participation_all",
    "inverse_participation_ratio_all",
    "EdgeSpectrumResult",
    "geometry_projection",
    "apply_geometry_to_hamiltonian",
    "GeometryProjection",
    "soft_dot_potential",
    "radial_mass_values",
    "apply_profiles_to_hamiltonian",
]
