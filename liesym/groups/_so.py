from __future__ import annotations

from typing import List, Literal, overload, Tuple, Union

from sympy import I, KroneckerDelta, Matrix, zeros

from ..algebras import B, D

from ._base import LieGroup


class SO(LieGroup):
    """The Special Orthogonal Group"""

    def __new__(cls, dim: int):
        if dim < 2:
            raise NotImplementedError("SO(1)==O(1) is not implemented yet.")
        return super().__new__(cls, "SO", dim)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        n = self.dimension
        if n.is_even:
            self._algebra = D(n / 2)
        else:
            self._algebra = B((n - 1) / 2)

    @overload
    def generators(
        self, *, cartan_only: bool = False, indexed: Literal[True]
    ) -> List[Tuple[Matrix, tuple]]:
        ...

    @overload
    def generators(
        self, cartan_only: bool = False, indexed: Literal[False] = False
    ) -> List[Matrix]:
        ...

    def generators(
        self, cartan_only: bool = False, indexed: bool = False
    ) -> Union[List[Matrix], List[Tuple[Matrix, tuple]]]:
        """Generators for SO(N).

        Args:
            cartan_only (bool, optional): Only return the cartan generators (diagonalized generators). Defaults to False.
            indexed (bool, Optional): For N > 3, there exists a naming scheme for generators. If True returns a tuple
            of the matrix and its (m,n) index.

        Returns:
            list[Union[Matrix, Tuple[Matrix, tuple]]]: list of (mathematical) generators

        Sources:
            - http://www.astro.sunysb.edu/steinkirch/books/group.pdf
        """
        return super().generators(cartan_only=cartan_only, indexed=indexed)

    def _calc_generator(self, indexed=False, **kwargs):
        results = []
        for m in range(self.dimension):
            for n in range(m):
                mat = zeros(self.dimension)
                for i in range(self.dimension):
                    for j in range(self.dimension):
                        mat[i, j] = -I * (
                            KroneckerDelta(m, i) * KroneckerDelta(n, j)
                            - KroneckerDelta(m, j) * KroneckerDelta(n, i)
                        )
                results.append((mat, (m, n)) if indexed else mat)
        if indexed:
            return sorted(results, key=lambda y: y[1])
        return results
