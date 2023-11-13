import os

try:
    from ._liesym_rust import (  # type: ignore[import-untyped]
        LieAlgebraBackend as _LieAlgebraBackend,
    )
except ImportError:
    # suppress import error during sphinx builds
    if os.environ.get("SPHINX_BUILD") != "1":
        raise
    else:
        _LieAlgebraBackend = ""

from .groups import *
from .algebras import *


# C++ backend for sympy. Symbolic speed ups
os.environ["USE_SYMENGINE"] = "1"

IS_SYMENGINE = os.environ.get("USE_SYMENGINE") == "1"


__all__ = [
    "A",
    "B",
    "C",
    "D",
    "F4",
    "G2",
    "E",
    "LieAlgebra",
    "NumericSymbol",
    "SU",
    "generalized_gell_mann",
    "SO",
    "Sp",
    "Group",
    "LieGroup",
    "Z",
    "root_angle",
]
