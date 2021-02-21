from sympy.core import Basic
from sympy.core.sympify import _sympify
from sympy import Matrix
from typing import List

from .methods import (
    _cartan_matrix,
    _cocartan_matrix,
    _quadratic_form,
    _reflection_matricies,
)

from .backend import create_backend


class LieAlgebra(Basic):
    def __new__(cls, series: str, rank: int):
        """Returns a new instance of a Sympy object

        Args:
            series (str): The series type
            rank (int): The rank of the algebra
        """
        return super().__new__(cls, series, _sympify(rank))

    def __init__(self, *args, **kwargs):
        """Used to set lazy properties
        """
        self._simple_roots = None
        self._positive_roots = None
        self._cartan_matrix = None
        self._omega_matrix = None
        self._quadratic_form = None
        self._cocartan_matrix = None
        self._reflection_matricies = None
        self._fundamental_weights = None
        self._backend = None

    @property
    def series(self) -> str:
        """Algebra series type
        """
        return self.args[0]

    @property
    def rank(self) -> int:
        """Algebra rank
        """
        return self.args[1]

    @property
    def dimension(self) -> int:
        """Algebra dimension
        """

    @property
    def roots(self) -> int:
        """Total number of roots in the algebra
        """

    @property
    def simple_roots(self) -> List[Matrix]:
        """Returns a list of Sympy matrix (1,dimension)
        objects representing a chosen basis of the algebra.

        This method can be overridden to choose your own basis,
        be sure to do this before any other properties are called
        as they are lazily evaluated and the simple_roots define
        the entire representation of the algebra.

        Examples
        ========
        ```python
        algebra = F4()

        my_simple_roots = [
                # my basis
            ]
        algebra.simple_roots = my_simple_roots
        ```
        """
        return self._simple_roots

    @simple_roots.setter
    def simple_roots(self, val: List[Matrix]):
        """Overrides the default representation of the algebras simple_roots
        """
        assert len(val) == len(self._simple_roots), "Incorrect number of simple roots"
        self._simple_roots = val

    @property
    def cartan_matrix(self) -> Matrix:
        if self._cartan_matrix is None:
            self._cartan_matrix = _cartan_matrix(self.simple_roots)
        return self._cartan_matrix

    @property
    def cocartan_matrix(self) -> Matrix:
        if self._cocartan_matrix is None:
            self._cocartan_matrix = _cocartan_matrix(self.simple_roots)
        return self._cocartan_matrix

    @property
    def omega_matrix(self) -> Matrix:
        if self._omega_matrix is None:
            self._omega_matrix = self.cocartan_matrix.pinv().T
        return self._omega_matrix

    @property
    def metric_tensor(self) -> Matrix:
        if self._quadratic_form is None:
            self._quadratic_form = _quadratic_form(
                self.cartan_matrix, self.simple_roots)
        return self._quadratic_form

    @property
    def reflection_matricies(self) -> List[Matrix]:
        if self._reflection_matricies is None:
            self._reflection_matricies = _reflection_matricies(
                self.simple_roots)
        return self._reflection_matricies

    @property
    def fundamental_weights(self):
        if self._fundamental_weights is None:
            self._fundamental_weights = [self.omega_matrix.row(
                i) for i in range(self.omega_matrix.rows)]
        return self._fundamental_weights

    def orbit(self, weight: Matrix, stabilizers=None, **kwargs) -> List[Matrix]:
        if self._backend is None:
            self._backend = create_backend(self)
        return self._backend.orbit(weight, stabilizers)

    def root_system(self, **kwargs) -> List[Matrix]:
        if self._backend is None:
            self._backend = create_backend(self)
        return self._backend.root_system()

