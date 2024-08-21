__all__ = [
    "TauModel",
    "TBTauModel",
]

from amoeba2.tau_model import TauModel
from amoeba2.tb_tau_model import TBTauModel

from . import _version

__version__ = _version.get_versions()["version"]
