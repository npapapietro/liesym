from __future__ import annotations

from sympy import Matrix, zeros

from ..algebras import C

from ._base import LieGroup


def _iachello_basis(dim):
    """Basis noted in Francesco Iachello's text."""
    n = dim / 2

    def E(i, j):
        mat = zeros(dim)
        mat[i, j] = 1
        return mat

    for k in range(n):
        yield E(k, n + k)
        yield E(k + n, k)

    for k in range(n):
        for m in range(n):
            yield E(k, m) - E(n + m, n + k)

    for m in range(n):
        for k in range(m):
            yield E(k, n + m) + E(m, n + k)
            yield E(n + k, m) + E(n + m, k)


class Sp(LieGroup):
    """The Symplectic Group"""

    def __new__(cls, dim: int):
        if dim % 2 != 0:
            raise NotImplementedError("Sp is not defined for odd dimensions.")
        return super().__new__(cls, "Sp", dim)

    def __init__(self, *args, **kwargs):
        self._algebra = C(self.dimension / 2)
        super().__init__(*args, **kwargs)

    def generators(self, cartan_only=False, **kwargs):
        """Generators for Sp(2N). There are a lot of possible choices, so
        we choose one based on existing literature for generators to
        be in Iachello's basis.

        Sources:
            - Iachello, F (2006). Lie algebras and applications. ISBN 978-3-540-36236-4.
        """
        return super().generators(cartan_only=cartan_only, **kwargs)

    def _calc_generator(self, **_):
        return list(_iachello_basis(self.dimension))
