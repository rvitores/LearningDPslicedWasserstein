# This code has been imported (and possibly modified) from Meta's OPACUS Privacy engine (https://github.com/pytorch/opacus) which was distributed under Apache 2.0 License.

from .accountant import IAccountant
from .gdp import GaussianAccountant
from .rdp import RDPAccountant


__all__ = [
    "IAccountant",
    "GaussianAccountant",
    "RDPAccountant",
]


def create_accountant(mechanism: str) -> IAccountant:
    if mechanism == "rdp":
        return RDPAccountant()
    elif mechanism == "gdp":
        return GaussianAccountant()

    raise ValueError(f"Unexpected accounting mechanism: {mechanism}")
