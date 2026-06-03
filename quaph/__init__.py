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
from quaph._diff import plot_diff
from quaph._run import (
    run_analytic,
    run_simulated_ideal,
    run_simulated_noisy,
    load_result,
    AnalyticResult,
    SimulatedResult,
)

__all__ = [
    "Model",
    "ModelCapabilityError",
    "get_model",
    "register_model",
    "register_model_from_file",
    "build_tight_binding_model",
    "remove_model",
    "run_analytic",
    "run_simulated_ideal",
    "run_simulated_noisy",
    "load_result",
    "AnalyticResult",
    "SimulatedResult",
    "plot_diff",
]
