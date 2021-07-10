from sympy import sympify, Basic, Matrix
from typing import List

from ._base import LieGroup
from ..algebras import A


class SU(LieGroup):

    def __init__(self, *args, **kwargs):
        dim = self.dimension - 1
        self._algebra = A(dim)

    def get_generators(self) -> List[Matrix]:
        pass