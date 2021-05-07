from sympy.core.sympify import _sympify
from sympy import Matrix

from ._base import LieAlgebra


def _euclidean_root(i, n):
    root = [0]*n
    root[i] = 1
    try:
        root[i+1] = -1
    except IndexError:
        pass  # catches B last root
    return Matrix([root])


class A(LieAlgebra):
    r"""The compact lie group of type A. The dynkin diagram for this algebra is

        .. image:: ../../docs/source/images/type_A.png
           :width: 300px
           :align: center

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
            n + 1
        """
        return self.rank + 1

    @property
    def roots(self) -> int:
        """The number of roots for the simple Lie algebra A is 
        defined as 

        .. math::
            n(n + 1)
        """
        return self.rank * (self.rank + 1)


class B(LieAlgebra):
    r"""The compact lie group of type B. The dynkin diagram for this algebra is

        .. image:: ../../docs/source/images/type_B.png
           :width: 300px
           :align: center

    """
    def __new__(cls, n):
        return super().__new__(cls, "B", _sympify(n))

    def __init__(self, *args, **kwargs):
        super().__init__()
        n = self.rank
        self._simple_roots = [_euclidean_root(i, n) for i in range(n)]

    @property
    def dimension(self) -> int:
        """The dimension of the simple Lie algebra B series is
        equal to the rank of the algebra.
        """

        return self.rank

    @property
    def roots(self) -> int:
        """The number of roots for the simple Lie algebra B is 
        defined as 

        .. math::
            2n^2
        """
        return 2 * self.rank**2


class C(LieAlgebra):
    r"""The compact lie group of type C. The dynkin diagram for this algebra is

        .. image:: ../../docs/source/images/type_C.png
           :width: 300px
           :align: center
    """

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
        """The dimension of the simple Lie algebra C series is
        equal to the rank of the algebra.
        """

        return self.rank

    @property
    def roots(self) -> int:
        """The number of roots for the simple Lie algebra C is 
        defined as 

        .. math::
            2n^2
        """

        return 2 * self.rank**2


class D(LieAlgebra):
    r"""The compact lie group of type D. The dynkin diagram for this algebra is

        .. image:: ../../docs/source/images/type_D.png
           :width: 300px
           :align: center
    """

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
        """The dimension of the simple Lie algebra D series is
        equal to the rank of the algebra.
        """

        return self.rank

    @property
    def roots(self) -> int:
        """The number of roots for the simple Lie algebra D is 
        defined as 

        .. math::
            2n(n-1)
        """

        return 2 * self.rank * (self.rank - 1)
