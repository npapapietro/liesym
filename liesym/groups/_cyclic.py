# from __future__ import annotations
from functools import reduce

from sympy import exp, I, pi, Symbol

from ._base import Group


class Z(Group):
    """Cyclic Group
    """

    def __new__(cls, dim: int):
        return super().__new__(cls, "Z", dim)

    def generators(self, indexed=False) -> list:
        """The basis for the generators in the cyclic group are 
        in the exponential imaginary basis.

        Args:
            indexed (bool, Optional): Returns tuple of named generators. Defaults to False.

        Examples
        =========
        >>> from liesym import Z
        >>> from sympy import I, pi
        >>> Z(3).generators()
        [1, exp(2*I*pi/3), exp(-2*I*pi/3)]
        >>> Z(3).generators(indexed=True)
        [(Z_0, 1), (Z_1, exp(2*I*pi/3)), (Z_2, exp(-2*I*pi/3))]
        """
        d = self.dimension

        gens = [exp(2 * pi * I * x / d) for x in range(d)]
        if indexed:
            return [(Symbol(f"Z_{idx}"), x) for idx, x in enumerate(gens)]
        return gens

    def product(self, *args, **kwargs) -> list:
        """Product of the Cyclic representations

        Returns:
            list: To keep types standard, returns a list of length 1

        Examples
        =========
        >>> from liesym import Z
        >>> z5 = Z(5)
        >>> g = z5.generators()
        >>> z5.product(g[0], g[1], g[2], g[3], g[4])
        [1]
        """
        return [reduce(lambda a, b: a*b, args)]

    def sym_product(self, *args, as_tuple=False, **kwargs) -> list:
        """Will take the product symbolically of Cyclic representations
        of the form `lambda x: f`Z_{x}`

        Returns:
            list: To keep types standard, returns a list of length 1

        Examples
        =========
        >>> from liesym import Z
        >>> z5 = Z(5)
        >>> z5.sym_product("Z_1", "Z_2")
        [Z_3]
        >>> z5.sym_product("Z_3", "Z_4", as_tuple=True)
        [(Z_2, exp(4*I*pi/5))]
        """

        cleaned_args = [Symbol(x) if isinstance(x, str) else x for x in args]

        gens = {Symbol(f"Z_{idx}"): x for idx,
                x in enumerate(self.generators())}

        result = self.product(*[gens[x] for x in cleaned_args])

        for k, v in gens.items():
            if v == result[0]:
                if as_tuple:
                    return [(k, v)]
                return [k]
        raise ValueError("Malformed cyclic product")
