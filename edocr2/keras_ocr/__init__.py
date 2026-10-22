# Submodules load when first used, so the parts that need no TensorFlow (tools, craft) can be
# imported without it; detection and recognition still bring it in when asked for.
import importlib

__version__ = "0.9.0"

_SUBMODULES = {"detection", "recognition", "tools", "data_generation", "pipeline", "evaluation", "datasets", "config", "craft"}


def __getattr__(name):
    if name in _SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
