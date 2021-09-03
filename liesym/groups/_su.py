from __future__ import annotations

from typing import Generator
from sympy import I, Basic
from sympy.core.backend import zeros, sqrt

from ._base import LieGroup
from ..algebras import A


def generalized_gell_mann(dimension: int) -> Generator:
    """Generator of generalized Gell-Mann Matrices. Note order may
    be different than usual because we generate symmetric, then antisymmetric, 
    then diagonal.

    Args:
        dimension (int): Dimension of group

    Yields:
        Generator[Matrix]: Python Generator of (mathematical) generators

    Sources:
        - https://mathworld.wolfram.com/GeneralizedGell-MannMatrix.html

    """
    def eij(dim):
        def _(i, j):
            mat = zeros(dim)
            mat[i, j] = 1
            return mat
        return _
    E = eij(dimension)

    # Symmetric
    for k in range(dimension):
        for j in range(k):
            yield E(k, j) + E(j, k)

    # AntiSymmetric
    for k in range(dimension):
        for j in range(k):
            yield I * (E(k, j) - E(j, k))

    # Diagonal
    for l in range(dimension - 1):
        coeff = sqrt(2) / sqrt((l+1) * (l + 2))
        sum_term = sum([E(j, j) for j in range(l+1)], zeros(dimension))
        yield coeff * (sum_term - (l+1) * E(l+1, l+1))


class SU(LieGroup):
    """The Special Unitary Group
    """

    def __new__(cls, dim: int):
        if dim < 2:
            raise NotImplementedError("SU(1)==U(1) is not implemented yet.")
        return super().__new__(cls, "SU", dim)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._algebra = A(self.dimension - 1)
        self._generators = None

    def generators(self):
        r"""Generators for SU(N).  Based off the generalized Gell-Mann matrices $\lambda_a$
        the generators, $T_a$ are

        .. math::

            T_a = \frac{\lambda_a}{2}            
        """
        if self._generators is None:
            # typical normalization
            self._generators = [
                x / 2 for x in generalized_gell_mann(self.dimension)]
        return self._generators
