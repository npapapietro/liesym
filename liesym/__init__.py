"""
Liesym is an extension module on SymPy that reimplements the
liealgebra module inside sympy with the idea of using a
compiled backend to do speedups on some of the slower aspects
of SymPy's pure python implementation. This python module is
mostly a reimplementation of a Mathematica module LieART.


* SymPy: https://sympy.org
* LieART: https://arxiv.org/pdf/1206.6379.pdf
"""

# Compiled extension imports
from .liesym import (
    debug_mode as _debug_mode,
    LieAlgebraBackend as _LieAlgebraBackend,
)

from .algebras import *

__all__ = [
    "A", "B", "C", "D", "F4", "G2", "E", "LieAlgebra", "NumericSymbol"
]
