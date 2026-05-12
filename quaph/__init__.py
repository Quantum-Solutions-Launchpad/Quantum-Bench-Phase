from quaph._core import setup_logging as _setup_logging
_setup_logging()

from quaph._model import Model, ModelCapabilityError
from quaph._registry import get_model, register_model, remove_model
from quaph._run import (
    run_analytic,
    run_simulated_ideal,
    run_simulated_noisy,
    load_result,
    AnalyticResult,
    SimulatedResult,
)
from quaph import models
from quaph.models import haldane, hubbard, haldane_hubbard

__all__ = [
    "Model",
    "ModelCapabilityError",
    "get_model",
    "register_model",
    "remove_model",
    "run_analytic",
    "run_simulated_ideal",
    "run_simulated_noisy",
    "load_result",
    "AnalyticResult",
    "SimulatedResult",
    "models",
    "haldane",
    "hubbard",
    "haldane_hubbard",
]
