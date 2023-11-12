from __future__ import annotations

from functools import reduce
from typing import List, Literal, overload, Tuple, Union

from sympy import conjugate, exp, I, pi, Symbol

from ._base import Group


class Z(Group):
    """Cyclic Group"""

    def __new__(cls, dim: int):
        return super().__new__(cls, "Z", dim)

    def __init__(self, *args, **kwargs):
        self._lookups = {
            Symbol(f"Z_{idx}"): x for idx, x in enumerate(self.generators())
        }

    @overload
    def generators(self, indexed: Literal[False] = False) -> List[exp]:
        ...

    @overload
    def generators(self, indexed: Literal[True]) -> List[Tuple[Symbol, exp]]:
        ...

    def generators(self, indexed: bool = False):
        """The basis for the generators in the cyclic group are
        in the exponential imaginary basis.

        Args:
            indexed (bool, Optional): Returns tuple of named generators. Defaults to False.

        Examples
        ========
        >>> from liesym import Z
        >>> from sympy import I, pi
        >>> Z(3).generators()
        [1, exp(2*I*pi/3), exp(-2*I*pi/3)]
        >>> Z(3).generators(indexed=True)
        [(Z_0, 1), (Z_1, exp(2*I*pi/3)), (Z_2, exp(-2*I*pi/3))]
        """
        d = self.dimension

        gens = [
            exp(2 * pi * I * x / d) for x in range(int(d))
        ]  # type:ignore[call-overload]
        if indexed:
            return [(Symbol(f"Z_{idx}"), x) for idx, x in enumerate(gens)]
        return gens

    @overload
    def product(self, as_tuple: Literal[False] = False) -> List[exp]:
        ...

    @overload
    def product(self, as_tuple: Literal[True]) -> List[Tuple[Symbol, exp]]:
        ...

    def product(self, *args: Union[str, Symbol], as_tuple=False):
        """Product of the Cyclic representations

        Args:
            args: A list of args to take product of
            as_tuple (bool, optional): Returns a tuple of their symbolic rep and exp val instead of just exp val

        Examples
        ========
        >>> from liesym import Z
        >>> z5 = Z(5)
        >>> g = z5.generators()
        >>> z5.product(g[0], g[1], g[2], g[3], g[4])
        [1]
        >>> z5.product("Z_1", "Z_2")
        [Z_3]
        >>> z5.product("Z_3", "Z_4", as_tuple=True)
        [(Z_2, exp(4*I*pi/5))]
        """
        if all([isinstance(x, Symbol) for x in args]):
            results = [reduce(lambda a, b: a * b, [self._lookups[x] for x in args])]
        elif all([isinstance(x, str) for x in args]):
            args = [Symbol(x) for x in args]  # type: ignore[assignment]
            results = self.product(*args)
        elif all([hasattr(x, "__mul__") for x in args]):
            results = [reduce(lambda a, b: a * b, args)]  # type: ignore
        else:
            raise TypeError("Unsupported type of args, please use string or symbol or exp")
        for k, v in self._lookups.items():
            if v == results[0] or k == results[0]:
                if as_tuple:
                    return [(k, v)]
                return [k]
        raise ValueError("Malformed cyclic product")

    def conjugate(self, rep, symbolic=False):
        """Finds the conjugate representation of the cyclic representation

        Examples
        ========
        >>> from liesym import Z
        >>> from sympy import sympify
        >>> z3 = Z(3)
        >>> g = z3.generators()
        >>> z3.conjugate(g[0])
        1
        >>> assert z3.conjugate("Z_1", symbolic=True) == sympify("Z_2")
        """
        if symbolic:
            cleaned_rep = Symbol(rep) if isinstance(rep, str) else rep
            math_rep = self._lookups[cleaned_rep]
            conj_rep = conjugate(math_rep)
            return [k for k, v in self._lookups.items() if v == conj_rep][0]

        if len([k for k, v in self._lookups.items() if v == rep]) == 0:
            raise KeyError("Rep not in cyclic group.")
        return conjugate(rep)

    def irrep_lookup(self, irrep):
        """Returns the symbol of irrep

        Examples
        ========
        >>> from liesym import Z
        >>> from sympy import sympify
        >>> z3 = Z(3)
        >>> z3.irrep_lookup("Z_1")
        exp(2*I*pi/3)
        """
        cleaned_rep = Symbol(irrep) if isinstance(irrep, str) else irrep
        return self._lookups[cleaned_rep]
