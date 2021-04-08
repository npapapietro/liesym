from sympy import Matrix, S
from sympy.core.sympify import _sympify

from .base import LieAlgebra


class F4(LieAlgebra):
    def __new__(cls):
        return super().__new__(cls, "F", _sympify(4))

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._simple_roots = [
            Matrix([[1, -1, 0, 0]]),
            Matrix([[0, 1, -1, 0]]),
            Matrix([[0, 0, 1, 0]]),
            Matrix([[-S.Half, -S.Half, -S.Half, -S.Half]]),
        ]

    @property
    def dimension(self) -> int:
        return 4

    @property
    def roots(self) -> int:
        return 48


class G2(LieAlgebra):
    def __new__(cls):
        return super().__new__(cls, "G", _sympify(2))

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._simple_roots = [
            Matrix([[0, 1, -1]]),
            Matrix([[1, -2, 1]])
        ]

    @property
    def dimension(self) -> int:
        return 3

    @property
    def roots(self) -> int:
        return 12


def _e_series_default_roots(n):
    e8 = [
        [S.Half, -S.Half, -S.Half,
         -S.Half, -S.Half, -S.Half,
         -S.Half, S.Half],
        [-1, 1, 0, 0, 0, 0, 0, 0],
        [0, -1, 1, 0, 0, 0, 0, 0],
        [0, 0, -1, 1, 0, 0, 0, 0],
        [0, 0, 0, -1, 1, 0, 0, 0],
        [0, 0, 0, 0, -1, 1, 0, 0],
        [0, 0, 0, 0, 0, -1, 1, 0],
        [1, 1, 0, 0, 0, 0, 0, 0]]

    roots = e8[:n - 1] + [e8[-1]]
    return [Matrix([roots[i]]) for i in range(n)]


class E(LieAlgebra):
    def __new__(cls, n):
        if n not in [6,7,8]:
            raise ValueError("Algebra series E only defined for 6, 7 and 8}")
        return super().__new__(cls, "E", _sympify(n))

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._simple_roots = _e_series_default_roots(self.args[1])

    @property
    def dimension(self) -> int:
        return self.rank

    @property
    def roots(self) -> int:
        return [72, 126, 240][self.rank-6]

