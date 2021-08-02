from __future__ import annotations

from typing import Generator, Union
from sympy import Matrix, zeros, I, sqrt, Basic, sympify, trace
from sympy.tensor.array.dense_ndim_array import MutableDenseNDimArray

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




def commutator(A: Basic, B: Basic, anti=False) -> Basic:
    r"""Performs commutation brackets on A,B.

    If anti is False

    .. math::
        [ A, B ] = A * B - B * A

    Otherwise 

    .. math::
        \{A, B\} = A * B + B * A

    Args:
        A (Basic): Any mathematical object
        B (Basic): Any mathematical object
        anti (bool, optional): Anticommutation. Defaults to False.

    Returns:
        Basic: The (anti)commutation bracket result
    """
    if anti:
        return A * B + B * A
    return A * B - B * A


class SU(LieGroup):
    """The Special Unitary Group
    """

    def __new__(cls, dim: int):
        if dim < 2:
            raise NotImplementedError("SU(1)==U(1) is not implemented yet.")
        return super().__new__(cls, "SU", dim)

    def __init__(self, *args, **kwargs):
        self._algebra = A(self.dimension - 1)
        self._generators = None
        self._structure_constants = (None, None)

    def generators(self):
        r"""Generators for SU(N).  Based off the generalized Gell-Mann matrices $\lambda_a$
        the generators, $T_a$ are

        .. math::

            T_a = \frac{\lambda_a}{2}            
        """
        if self._generators is None:
            # typical normalization
            self._generators = [x / 2 for x in generalized_gell_mann(self.dimension)]
        return self._generators

    def structure_constants(self, *idxs: int) -> Union[Basic, Matrix]:
        """Returns the structure constants of the group. Indexes start at 0 and constants maybe
        in different orderings than existing literature, but will still be in the Gell-Mann basis.
        Structure constants $f_{abc}$ is defined as

        .. math::
            [T_a, T_b] = \sum_c f_{abc} T_c

        where the group generators are $T$.

        Args:
            idxs (int): Optional postional arguments of 3 indices to return structure constant. If omitted, will return array.

        Returns:
            Union[Basic, Matrix]: If indicies are passed in, will return corresponding
            structure constant. Otherwise returns the 3dArry for entire group.
        """

        if self._structure_constants == (None, None):
            self._structure_constants = self._calculate_structure_constants()

        if len(idxs) > 0:
            if len(idxs) != 3:
                raise ValueError(
                    "3 indices need to be passed in if calling `structure_constants` with indices")
            else:
                [i, j, k] = idxs
                return self._structure_constants[0][i, j, k]
        return self._structure_constants[0]

    def _calculate_structure_constants(self):
        """Calculates the structure constants"""
        gens = self.generators()
        n = len(gens)

        f = MutableDenseNDimArray.zeros(n, n, n) * sympify("0")
        d = MutableDenseNDimArray.zeros(n, n, n) * sympify("0")

        for i in range(n):
            for j in range(n):
                for k in range(n):
                    f[i, j, k] = -2 * I * trace(commutator(gens[i],gens[j]) * gens[k])
                    d[i, j, k] = 2 * trace(commutator(gens[i],gens[j], anti=True) * gens[k])
        return (f, d)

    def d_coeffecients(self, *idxs: int):
        if self._structure_constants == (None, None):
            self._structure_constants = self._calculate_structure_constants()

        if len(idxs) > 0:
            if len(idxs) != 3:
                raise ValueError(
                    "3 indices need to be passed in if calling `structure_constants` with indices")
            else:
                [i, j, k] = idxs
                return self._structure_constants[1][i, j, k]
        return self._structure_constants[1]
