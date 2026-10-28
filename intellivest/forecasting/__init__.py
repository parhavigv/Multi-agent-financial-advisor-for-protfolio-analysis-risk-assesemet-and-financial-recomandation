"""Leakage-safe forecasting models, cross-validation and evaluation.

Import convention::

    from intellivest.forecasting import evaluate, build_panel, HMMRegime

Public names are resolved lazily (PEP 562) so that lightweight callers, such
as the market-data providers that only need :func:`infer_regime`, do not pull
in the full evaluation harness, the backtester and the agent stack. Importing
a submodule still executes this file, so eager imports here would make
``intellivest.data.real`` depend on the evaluation package.
"""

from __future__ import annotations

import importlib
from typing import Any

__version__ = "0.2.0"

# public name -> module that defines it
_LAZY: dict[str, str] = {
    "ForecastModel": ".base",
    "PanelData": ".base",
    "build_features": ".data",
    "build_panel": ".data",
    "forward_return": ".data",
    "forward_volatility": ".data",
    "EvaluationResult": ".evaluation",
    "evaluate": ".evaluation",
    "render_markdown": ".evaluation",
    "write_report": ".evaluation",
    "HMMRegime": ".regime",
    "infer_regime": ".regime",
    "BASELINES": ".return_models",
    "CANDIDATES": ".return_models",
    "ElasticNetForecast": ".return_models",
    "LightGBMForecast": ".return_models",
    "MomentumForecast": ".return_models",
    "OneOverNForecast": ".return_models",
    "RidgeForecast": ".return_models",
    "ShrunkMeanForecast": ".return_models",
    "EWMAVolatility": ".vol_models",
    "GARCH11Volatility": ".vol_models",
    "HARVolatility": ".vol_models",
}

__all__ = sorted(_LAZY)


def __getattr__(name: str) -> Any:
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))
