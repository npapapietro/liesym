from __future__ import annotations

from typing import List, Tuple, Union
from sympy import Matrix, zeros, I, KroneckerDelta

from ._base import LieGroup
from ..algebras import B, D



class SO(LieGroup):
    """The Special Orthogonal Group
    """

    def __new__(cls, dim: int):
        if dim < 2:
            raise NotImplementedError("SO(1)==O(1) is not implemented yet.")
        return super().__new__(cls, "SO", dim)

    def __init__(self, *args, **kwargs):
        n = self.dimension
        if n.is_even:
            self._algebra = D(n / 2)
        else:
            self._algebra = B((n-1) / 2)


    def generators(self, indexed=False) -> List[Union[Matrix, Tuple[Matrix, tuple]]]:
        """Generators for SO(N).

        Args:
            indexed (bool, Optional): For N > 3, there exists a naming scheme for generators. If True returns a tuple
            of the matrix and its (m,n) index.

        Returns:
            List[Union[Matrix, Tuple[Matrix, tuple]]]: List of (mathematical) generators

        Sources:
            - http://www.astro.sunysb.edu/steinkirch/books/group.pdf
        """
        results = []
        for m in range(self.dimension):
            for n in range(m):                      
                mat = zeros(self.dimension)          
                for i in range(self.dimension):
                    for j in range(self.dimension):
                        mat[i,j] = - I * (KroneckerDelta(m,i) * KroneckerDelta(n,j) - KroneckerDelta(m,j) * KroneckerDelta(n,i))
                results.append((mat, (m,n)) if indexed else mat)
        if indexed:
            return sorted(results, key=lambda y: y[1])
        return results