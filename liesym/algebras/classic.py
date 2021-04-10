from sympy.core.sympify import _sympify
from sympy import Matrix

from .base import LieAlgebra


def _euclidean_root(i, n):
    root = [0]*n
    root[i] = 1
    try:
        root[i+1] = -1
    except IndexError: pass # catches B last root
    return Matrix([root])


class A(LieAlgebra):
    """The compact lie group of type A has a dynkin diagram of

    .. raw:: latex

        A/{}

    """
    def __new__(cls, n):
        return super().__new__(cls, "A", _sympify(n))

    def __init__(self, *args, **kwargs):
        super().__init__()
        n = self.rank
        self._simple_roots = [_euclidean_root(i, n+1) for i in range(n)]

    @property
    def dimension(self) -> int:
        """The dimension of the simple Lie algebra A series is
        one greater than the rank of the algebra.

        .. math::
            dim = n + 1
        """
        return self.rank + 1

    @property
    def roots(self) -> int:
        """The number of roots for the simple Lie algebra A is 
        defined as 

        .. math::
            n^2 + 1
        """
        return self.rank*(self.rank + 1)


class B(LieAlgebra):
    def __new__(cls, n):
        return super().__new__(cls, "B", _sympify(n))

    def __init__(self, *args, **kwargs):
        super().__init__()
        n = self.rank
        self._simple_roots = [_euclidean_root(i, n) for i in range(n)]

    @property
    def dimension(self) -> int:
        return self.rank

    @property
    def roots(self) -> int:
        return 2 * self.rank**2


class C(LieAlgebra):
    def __new__(cls, n):
        return super().__new__(cls, "C", _sympify(n))

    def __init__(self, *args, **kwargs):
        super().__init__()
        n = self.rank
        c_root = [0] * n
        c_root[-1] = 2
        self._simple_roots = [_euclidean_root(i, n) for i in range(
            n - 1)] + [Matrix([c_root])]

    @property
    def dimension(self) -> int:
        return self.rank

    @property
    def roots(self) -> int:
        return 2 * self.rank**2


class D(LieAlgebra):
    """test"""
    def __new__(cls, n):
        return super().__new__(cls, "D", _sympify(n))

    def __init__(self, *args, **kwargs):
        super().__init__()
        n = self.rank
        d_root = [0] * n
        d_root[-1] = 1
        d_root[-2] = 1
        self._simple_roots = [_euclidean_root(i, n) for i in range(
            n - 1)] + [Matrix([d_root])]

    @property
    def dimension(self) -> int:
        return self.rank

    @property
    def roots(self) -> int:
        return 2 * self.rank * (self.rank - 1)
