from sympy import exp, I, simplify, conjugate, log, sympify, Symbol, expand_log
from functools import reduce

from ._base import Group


class U1(Group):
    """U1 Group. Technically continuous, but represented
    in this module as discrete"""

    def __new__(cls):
        return super().__new__(cls, "U", 1)

    def __init__(self, *args, **kwargs):
        self._theta = Symbol(r"\theta", real=True)

    def generators(self) -> list:
        """Infinite circle group generators
        """
        return [exp(I * self._theta)]

    def _from_charge(self, q):
        return exp(I * self._theta * q)

    def product(self, *args, **kwargs) -> list:
        """Sums up all the charges"""
        return [simplify(reduce(lambda a, b: a*b, args))]

    def sym_product(self, *args, as_tuple=False, **kwargs) -> list:
        """Sums up all the charges, symbolically"""

        result = self.product(*[self._from_charge(x) for x in args], **kwargs)

        if as_tuple:
            return [(self._from_charge(x), x) for x in result]
        return result

    def conjugate(self, rep, symbolic=False, **kwargs):
        r"""Finds the conjugate representation of the U1 representation

        Examples
        =========
        >>> from liesym import U1
        >>> from sympy import *
        >>> u1 = U1()
        >>> u1.conjugate("1/6", symbolic=True)
        -1/6
        >>> print(u1.irrep_lookup("1/6"))
        exp(I*\theta/6)
        """
        if symbolic:
            cleaned_rep = sympify(rep) if isinstance(rep, str) else rep
            math_rep = self._from_charge(cleaned_rep)
            return expand_log(log(conjugate(math_rep)), force=True) / (I * self._theta)

        return conjugate(rep)

    def irrep_lookup(self, irrep):
        cleaned_rep = sympify(irrep) if isinstance(irrep, str) else irrep
        return self._from_charge(cleaned_rep)

    @staticmethod
    def dynkin_index(charge):
        """Returns the dynkin index for the abelian gauge"""
        return charge**2
