"""Models module for AQI Prediction System."""

# Lazy imports to avoid RuntimeWarning when running `python -m models.trainer`
def __getattr__(name):
    if name == "ModelFactory":
        from models.trainer import ModelFactory
        return ModelFactory
    elif name == "TensorFlowNN":
        from models.trainer import TensorFlowNN
        return TensorFlowNN
    elif name == "HopsworksIntegration":
        from models.registry import HopsworksIntegration
        return HopsworksIntegration
    elif name == "LocalModelRegistry":
        from models.registry import LocalModelRegistry
        return LocalModelRegistry
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    return __all__

__all__ = [
    "ModelFactory",
    "TensorFlowNN",
    "HopsworksIntegration",
    "LocalModelRegistry",
]
