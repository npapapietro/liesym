from sympy import sympify, Basic, Matrix
from typing import List

from ..algebras import LieAlgebra

class LieGroup(Basic):
    def __new__(cls, series: str, dim: int):
        """
        Returns a new instance of a Sympy object

        Args:
            series (str): The series type
            dim (int): The dimension of the group
        """
        return super().__new__(cls, series, sympify(dim))

    def __init__(self, *args, **kwargs):
        """Used to set lazy properties
        """
        self._algebra = None

    @property
    def group(self) -> str:
        """Group type
        """
        return self.args[0]

    @property
    def dimension(self) -> int:
        """Group dimension
        """
        return self.args[1]

    @property
    def algebra(self) -> LieAlgebra:
        """Returns the underlying Lie Algebra"""
        return self._algebra

    def get_generators(self) -> List[Matrix]:
        pass