from __future__ import annotations

from typing import List, Tuple, Union, Any
from sympy.core import Basic
from sympy import Matrix, Symbol, sympify

from ..algebras import LieAlgebra


class Group(Basic):
    """The base class for all (lie) groups. The methods and properties
    in this class are basis independent and apply in a general sense. In
    order to write down the roots as matricies and vectors, we choose a 
    representation.
    """

    def __new__(cls, group: str, dim: int):
        obj = super().__new__(cls, sympify(dim))
        obj._group = group
        return obj

    @property
    def group(self) -> str:
        """Group type
        """
        return self._group

    @property
    def dimension(self) -> int:
        """Group dimension
        """
        return self.args[0]

    def generators(self) -> list:
        """Generalized matrix generators of the group.

        Abstract
        """
        pass

    def product(self, *args, **kwargs) -> list:
        """Calculates the product between several group representations.
        If backing algebra is a Lie Algebra, defaults to `LieAlgebra.tensor_product_decomposition`.

        Abstract
        """
        pass

    def sym_product(self, *args, as_tuple=False, **kwargs) -> list:
        """Calculates the product between several group representations passed by the symbolic name.
        If backing algebra is a Lie Algebra, defaults to `LieAlgebra.tensor_product_decomposition`

        Abstract
        """
        pass

    def conjugate(self, rep, symbolic=False):
        """Returns the conjugate representation. If the incoming rep is symbolic/named then it will return as such.

        Abstract
        """

    def irrep_lookup(self, irrep):
        """Returns the mathematical representation to the common name. Example would be returning `3` in SU(3) as `Matrix([[1,0]])`
        
        Abstract
        """


class LieGroup(Group):
    """Group that has a Lie Algebra associated with it.
    """

    def __init__(self, *args, **kwargs):
        """Used to set lazy properties
        """
        self._algebra = None

    @property
    def algebra(self) -> LieAlgebra:
        """Backing Lie Algebra
        """
        return self._algebra

    def product(self, *args, **kwargs) -> List['Matrix']:
        """Uses tensor product decomposition to find the products between the 
        representations. Supported kwargs can be found on `LieAlgebra.tensor_product_decomposition`

        Returns:
            List[Matrix]: Tensor sum of the product.

        Examples
        =========
        >>> from liesym import SO
        >>> from sympy import Matrix
        >>> so10 = SO(10)
        >>> so10.product(Matrix([[1,0,0,0,0]]),Matrix([[1,0,0,0,0]]))
        [Matrix([[0, 0, 0, 0, 0]]), Matrix([[0, 1, 0, 0, 0]]), Matrix([[2, 0, 0, 0, 0]])]
        """

        # type: ignore
        return self.algebra.tensor_product_decomposition(args, **kwargs)

    def sym_product(self, *args, as_tuple=False, **kwargs) -> List[Union[Symbol, Tuple[Matrix, Symbol]]]:
        r"""Uses tensor product decomposition to find the products between the 
        representations. Supported kwargs can be found on `LieAlgebra.tensor_product_decomposition`.

        Args:
            as_tuple (bool, optional): If True, returns tuple of matrix rep with symbol. Defaults to False.

        Returns:
            List[Union[Symbol, Tuple[Matrix, Symbol]]]: List of symbols or list of (Matrix, Symbol)

        Examples
        ========
        >>> from liesym import SO
        >>> so10 = SO(10)
        >>> so10.sym_product("10","45")        
        [120, 10, 320]
        >>> so10.sym_product("10","45", as_tuple=True)
        [(Matrix([[0, 0, 1, 0, 0]]), 120), (Matrix([[1, 0, 0, 0, 0]]), 10), (Matrix([[1, 1, 0, 0, 0]]), 320)]
        >>> from liesym import SU
        >>> su3 = SU(3)
        >>> su3.sym_product('3', r'\bar{3}')
        [1, 8]
        """
        mats = [self.algebra.irrep_lookup(x) for x in args]

        results = []
        for rep in self.algebra.tensor_product_decomposition(mats, **kwargs):
            symbol = self.algebra.dim_name(rep)
            if as_tuple:
                results.append((rep, symbol))
            else:
                results.append(symbol)
        return results

    def conjugate(self, rep, symbolic=False):
        r"""Uses the underlying algebra to find the conjugate representation.

        Examples
        =========
        >>> from liesym import SU
        >>> from sympy import Matrix
        >>> su3 = SU(3)
        >>> su3.conjugate(Matrix([[1,0]]))
        Matrix([[0, 1]])
        >>> su3.conjugate(3, symbolic=True) # sympy prints without quotes
        \bar{3}
        """
        if symbolic:
            rep = str(rep)
            math_rep = self.algebra.irrep_lookup(rep)
            conj_rep = self.algebra.conjugate(math_rep)
            return self.algebra.dim_name(conj_rep)
        return self.algebra.conjugate(rep)

    def irrep_lookup(self, irrep: str) -> Matrix:
        """Uses the underlying algebra to do a lookup on the common name to find the matrix representation."""
        return self.algebra.irrep_lookup(irrep)
