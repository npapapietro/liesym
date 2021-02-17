
from typing import List
from numpy import isnan
from sympy.core import Basic
from sympy import Matrix, S
from sympy.core.sympify import _sympify

from .methods import (
    _cartan_matrix,
    _cocartan_matrix,
    _quadratic_form,
    _reflection_matricies,
)

from .backend import orbit

class F4(Basic):
    def __new__(cls):
        return super().__new__(cls, "F", _sympify(4))

    def __init__(self, *args, **kwargs):
        self._simple_roots = None
        self._positive_roots = None
        self._cartan_matrix = None
        self._omega_matrix = None
        self._quadratic_form = None
        self._cocartan_matrix = None
        self._reflection_matricies = None
        self._fundamental_weights = None

    @property
    def series(self) -> str:
        return self.args[0]

    @property
    def rank(self) -> int:
        return self.args[1]

    @property
    def dimension(self) -> int:
        return 4

    @property
    def roots(self) -> int:
        return 48

    @property
    def simple_roots(self) -> List[Matrix]:
        if self._simple_roots is None:
            self._simple_roots = [
                Matrix([[1, -1, 0, 0]]),
                Matrix([[0, 1, -1, 0]]),
                Matrix([[0, 0, 1, 0]]),
                Matrix([[-S.Half, -S.Half, -S.Half, -S.Half]]),
            ]

        return self._simple_roots

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
    def funamdental_weights(self):
        if self._fundamental_weights is None:
            self._fundamental_weights = [self.omega_matrix.row(
                i) for i in range(self.omega_matrix.rows)]
        return self._fundamental_weights

    def orbit(self, weight: Matrix, stabilizers=None, **kwargs) -> List[Matrix]:
        orbit(self, weight, stabilizers=stabilizers, **kwargs)


    # def tensor_product_decomp(self, *irreps: Matrix, as_dims=False) -> List[Matrix]:

    #     return _tensor_decomp(
    #         self.simple_roots,
    #         self.cartan_matrix,
    #         self.cartan_matrix.pinv(),
    #         self.cocartan_matrix.T,
    #         self.omega_matrix,
    #         self.omega_matrix.pinv(),
    #         self.roots,
    #         self.rank,
    #         irreps,
    #     )
