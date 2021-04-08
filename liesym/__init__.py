"""
Liesym is an extension module on SymPy that reimplements the
liealgebra module inside sympy with the idea of using a
compiled backend to do speedups on some of the slower aspects
of SymPy's pure python implementation. The calculations are
implemented from a Mathematica package called LieART, which
perform much better under Computer Algebra, but suffer from
being a proprietary language (and a niche language).

* SymPy: https://sympy.org
* LieART: https://arxiv.org/pdf/1206.6379.pdf
"""

# Compiled extension imports
from .liesym import (
    debug_mode as _debug_mode,
    LieAlgebraBackend as _LieAlgebraBackend,
)

from .algebras import (A, B, C, D, F4, G2, E)

__all__ = [
    "A", "B", "C", "D", "F4", "G2", "E"
]