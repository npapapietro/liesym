from typing import List, Optional

from sympy import flatten, Matrix, S
from sympy.core.sympify import _sympify

from ._base import LieAlgebra


class F4(LieAlgebra):
    r"""The compact lie group of type F4. The dynkin diagram for this algebra is

    .. image:: ../../docs/source/images/type_F4.png
       :height: 50px
       :align: center

    """

    def __new__(cls, simple_roots: Optional[List[Matrix]] = None):
        """Creates the F4 algebra
        Args:
            simple_roots (list[Matrix] | None, optional): Overrides default simple roots. Use this for
            calculating the algebra in a different basis. Defaults to None.

        """
        return super().__new__(
            cls,
            "F",
            _sympify(4),
            simple_roots
            or [
                Matrix([[1, -1, 0, 0]]),
                Matrix([[0, 1, -1, 0]]),
                Matrix([[0, 0, 1, 0]]),
                Matrix([[-S.Half, -S.Half, -S.Half, -S.Half]]),
            ],
        )

    @property
    def dimension(self) -> int:
        return 4

    @property
    def n_pos_roots(self) -> int:
        return 24

    def max_dynkin_digit(self, irrep: Matrix) -> int:
        """Returns the max Dynkin Digit for the representation"""
        l = flatten(irrep.tolist())
        return max(l) + 1


class G2(LieAlgebra):
    r"""The compact lie group of type G2. The dynkin diagram for this algebra is

    .. image:: ../../docs/source/images/type_G2.png
       :height: 50px
       :align: center

    """

    def __new__(cls, simple_roots: Optional[List[Matrix]] = None):
        """Creates the G2 algebra
        Args:
            simple_roots (list[Matrix] | None, optional): Overrides default simple roots. Use this for
            calculating the algebra in a different basis. Defaults to None.

        """
        return super().__new__(
            cls,
            "G",
            _sympify(2),
            simple_roots or [Matrix([[0, 1, -1]]), Matrix([[1, -2, 1]])],
        )

    @property
    def dimension(self) -> int:
        return 3

    @property
    def n_pos_roots(self) -> int:
        return 6

    def max_dynkin_digit(self, irrep: Matrix) -> int:
        """Returns the max Dynkin Digit for the representation"""
        l = flatten(irrep.tolist())
        return max(l) + 3


def _e_series_default_roots(n):
    e8 = [
        [S.Half, -S.Half, -S.Half, -S.Half, -S.Half, -S.Half, -S.Half, S.Half],
        [-1, 1, 0, 0, 0, 0, 0, 0],
        [0, -1, 1, 0, 0, 0, 0, 0],
        [0, 0, -1, 1, 0, 0, 0, 0],
        [0, 0, 0, -1, 1, 0, 0, 0],
        [0, 0, 0, 0, -1, 1, 0, 0],
        [0, 0, 0, 0, 0, -1, 1, 0],
        [1, 1, 0, 0, 0, 0, 0, 0],
    ]

    roots = e8[: n - 1] + [e8[-1]]
    return [Matrix([roots[i]]) for i in range(n)]


class E(LieAlgebra):
    r"""The compact lie group of type E. There are only three defined for
    type E.


        .. figure:: ../../docs/source/images/type_E6.png
           :height: 100px
           :align: center

           E6


        .. figure:: ../../docs/source/images/type_E7.png
           :height: 100px
           :align: center

           E7


        .. figure:: ../../docs/source/images/type_E8.png
           :height: 100px
           :align: center

           E8

    """

    def __new__(cls, n: int, simple_roots: Optional[List[Matrix]] = None):
        """Creates the E algebra of rank `n`

        Args:
            n (int): Dimension of algbera
            simple_roots (list[Matrix] | None, optional): Overrides default simple roots. Use this for
            calculating the algebra in a different basis. Defaults to None.

        """
        if n not in [6, 7, 8]:
            raise ValueError("Algebra series E only defined for 6, 7 and 8")

        return super().__new__(
            cls, "E", _sympify(n), simple_roots or _e_series_default_roots(n)
        )

    @property
    def dimension(self) -> int:
        return self.rank

    @property
    def n_pos_roots(self) -> int:
        return [36, 63, 120][self.rank - 6]

    def max_dynkin_digit(self, irrep: Matrix) -> int:
        """Returns the max Dynkin Digit for the representation"""
        l = flatten(irrep).tolist()
        if self.rank == 6:
            return max(l) + 3
        return max(l)

    def _congruency_class(self, irrep):
        n = self.rank
        if n == 8:
            return 0

        l = flatten(irrep.tolist())
        if n == 7:
            return (l[3] + l[5] + l[6]) % 2

        if n == 6:
            return (l[0] - l[1] + l[3] - l[4]) % 3
