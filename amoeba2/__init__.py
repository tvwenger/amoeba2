__all__ = [
    "AbsorptionModel",
    "EmissionAbsorptionModel",
]

from amoeba2.absorption_model import AbsorptionModel
from amoeba2.emission_absorption_model import EmissionAbsorptionModel

from . import _version

__version__ = _version.get_versions()["version"]
