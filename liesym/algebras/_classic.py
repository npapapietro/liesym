from sympy.core.sympify import _sympify
from sympy import Matrix, flatten

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
    def n_pos_roots(self) -> int:
        """The number of positive roots for the simple Lie algebra A is 
        defined as 

        .. math::
            n(n + 1) / 2
        """
        return self.rank * (self.rank + 1) / 2

    def dim_name(self, irrep: Matrix) -> str:
        r"""Returns a sympy formatted symbol for the irrep.
        This is commonly used in physics literature"""
        if self.rank == 2:
            if irrep == Matrix([[2, 0]]):
                return self._dim_name_fmt(6)
            elif irrep == Matrix([[0, 2]]):
                return self._dim_name_fmt(6, True)
        return super().dim_name(irrep)

    def max_dynkin_digit(self, irrep: Matrix) -> int:
        """Returns the max Dynkin Digit for the representation"""
        l = flatten(irrep.tolist())
        return max(l) + int(len(l) < 5)

    def _congruency_class(self, irrep):
        r = self.rank+1
        v = int(
            Matrix([range(1, r)]).dot(irrep)
        )
        return v % r


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

    def max_dynkin_digit(self, irrep: Matrix) -> int:
        """Returns the max Dynkin Digit for the representation"""
        l = flatten(irrep.tolist())
        return max(l) + int(len(l) < 5)

    @property
    def dimension(self) -> int:
        """The dimension of the simple Lie algebra B series is
        equal to the rank of the algebra.
        """

        return self.rank

    @property
    def n_pos_roots(self) -> int:
        """The number of positive roots for the simple Lie algebra B is 
        defined as 

        .. math::
            n^2
        """
        return self.rank**2

    def _congruency_class(self, irrep):
        return flatten(irrep.tolist())[-1] % 2


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

    def max_dynkin_digit(self, irrep: Matrix) -> int:
        """Returns the max Dynkin Digit for the representation"""
        l = flatten(irrep.tolist())
        return max(l) + int(len(l) < 5)

    @property
    def dimension(self) -> int:
        """The dimension of the simple Lie algebra C series is
        equal to the rank of the algebra.
        """

        return self.rank

    @property
    def n_pos_roots(self) -> int:
        """The number of positive roots for the simple Lie algebra C is 
        defined as 

        .. math::
            n^2
        """

        return self.rank**2

    def _congruency_class(self, irrep):
        return sum(flatten(irrep.tolist())[::2]) % 2


class D(LieAlgebra):
    r"""The compact lie group of type D. The dynkin diagram for this algebra is

        .. image:: ../../docs/source/images/type_D.png
           :width: 300px
           :align: center
    """

    def __new__(cls, n):
        if n < 2:
            raise ValueError("Min dimension for type 'D' is 2")
        return super().__new__(cls, "D", _sympify(n))

    def __init__(self, *args, **kwargs):
        super().__init__()
        n = self.rank
        d_root = [0] * n
        d_root[-1] = 1
        d_root[-2] = 1
        self._simple_roots = [_euclidean_root(i, n) for i in range(
            n - 1)] + [Matrix([d_root])]

    def max_dynkin_digit(self, irrep: Matrix) -> int:
        """Returns the max Dynkin Digit for the representation"""
        l = flatten(irrep.tolist())
        return max(l) + int(len(l) < 5)

    @property
    def dimension(self) -> int:
        """The dimension of the simple Lie algebra D series is
        equal to the rank of the algebra.
        """

        return self.rank

    @property
    def n_pos_roots(self) -> int:
        """The number of roots for the simple Lie algebra D is 
        defined as 

        .. math::
            n(n-1)
        """

        return self.rank * (self.rank - 1)

    def _congruency_class(self, irrep):
        l = flatten(irrep.tolist())
        i = sum(l[-2:]) % 2
        n = self.rank
        j = sum(l[:-2][::2]) + (n-2) * l[-2] + n * l[-1]

        return (i, j)
