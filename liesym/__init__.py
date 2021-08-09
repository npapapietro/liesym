import os

# C++ backend for sympy. Symbolic speed ups
os.environ["USE_SYMENGINE"] = "1"

# Compiled extension imports
from .liesym import (
    debug_mode as _debug_mode,
    LieAlgebraBackend as _LieAlgebraBackend,
    structure_constants as _structure_constants_backend
)

from .algebras import *
from .groups import *

__all__ = [
    "A", "B", "C", "D", "F4", "G2", "E", "LieAlgebra", "NumericSymbol",
    "SU", "generalized_gell_mann", "SO", "Sp", "Group", "LieGroup", "Z"
]
