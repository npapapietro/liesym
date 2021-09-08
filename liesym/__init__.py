from .liesym import (
    LieAlgebraBackend as _LieAlgebraBackend,
)
from .groups import *
from .algebras import *
import os

# C++ backend for sympy. Symbolic speed ups
os.environ["USE_SYMENGINE"] = "1"

IS_SYMENGINE = (os.environ.get("USE_SYMENGINE") == "1")
# Compiled extension imports


__all__ = [
    "A", "B", "C", "D", "F4", "G2", "E", "LieAlgebra", "NumericSymbol",
    "SU", "generalized_gell_mann", "SO", "Sp", "Group", "LieGroup", "Z"
]
