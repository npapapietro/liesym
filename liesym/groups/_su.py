from __future__ import annotations

from typing import Iterator
from sympy import Matrix, zeros, I, sqrt

from ._base import LieGroup
from ..algebras import A


def generalized_gell_mann(dimension: int) -> Iterator[Matrix]:
    """Generator of generalized Gell-Mann Matrices.

    Args:
        dimension (int): Dimension of group

    Yields:
        Iterator[Matrix]: List of (mathematical) generators

    Sources:
        - https://mathworld.wolfram.com/GeneralizedGell-MannMatrix.html

    """
    def eij(dim):    
            def _(i,j):
                mat = zeros(dim)
                mat[i,j] = 1
                return mat
            return _
    E = eij(dimension)

    # Symmetric
    for k in range(dimension):
        for j in range(k):
            yield E(k,j) + E(j,k)

    # AntiSymmetric
    for k in range(dimension):
        for j in range(k):
            yield I * (E(k,j) - E(j,k))

    # Diagonal
    for l in range(dimension - 1):
        coeff =  sqrt(2) / sqrt((l+1) * (l + 2))
        sum_term =  sum([E(j,j) for j in range(l+1)], zeros(dimension))
        yield coeff * (sum_term - (l+1) * E(l+1, l+1))
    

class SU(LieGroup):
    """The Special Unitary Group
    """

    def __new__(cls, dim: int):
        if dim < 2:
            raise NotImplementedError("SU(1)==U(1) is not implemented yet.")
        return super().__new__(cls, "SU", dim)

    def __init__(self, *args, **kwargs):
        self._algebra = A(self.dimension - 1)

    def generators(self):
        """ Generators for SU(N) in the basis of Gell-Mann
        """
        return list(generalized_gell_mann(self.dimension))

