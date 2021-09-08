from __future__ import annotations

from typing import Tuple, Union
from sympy import Symbol, Basic, I, Basic, trace, sympify, Matrix
from sympy.core.backend import Matrix as _CMatrix
from sympy.tensor.array.dense_ndim_array import MutableDenseNDimArray


from ..algebras import LieAlgebra


def commutator(A: Basic, B: Basic, anti=False) -> Basic:
    r"""Performs commutation brackets on A,B.

    If anti is False

    .. math::
        [ A, B ] = A * B - B * A

    Otherwise 

    .. math::
        \{A, B\} = A * B + B * A

    Args:
        A (Basic): Any mathematical object
        B (Basic): Any mathematical object
        anti (bool, optional): Anticommutation. Defaults to False.

    Returns:
        Basic: The (anti)commutation bracket result
    """
    if anti:
        return A * B + B * A
    return A * B - B * A


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
        self._structure_constants = (None, None)

    @property
    def algebra(self) -> LieAlgebra:
        """Backing Lie Algebra
        """
        return self._algebra

    def product(self, *args, **kwargs) -> list['Matrix']:
        """Uses tensor product decomposition to find the products between the 
        representations. Supported kwargs can be found on `LieAlgebra.tensor_product_decomposition`

        Returns:
            list[Matrix]: Tensor sum of the product.

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

    def sym_product(self, *args, as_tuple=False, **kwargs) -> list[Union[Symbol, Tuple[Matrix, Symbol]]]:
        r"""Uses tensor product decomposition to find the products between the 
        representations. Supported kwargs can be found on `LieAlgebra.tensor_product_decomposition`.

        Args:
            as_tuple (bool, optional): If True, returns tuple of matrix rep with symbol. Defaults to False.

        Returns:
            list[Union[Symbol, Tuple[Matrix, Symbol]]]: list of symbols or list of (Matrix, Symbol)

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

    def structure_constants(self, *idxs: int) -> Union[Basic, Matrix]:
        r"""Returns the structure constants of the group. Indexes start at 0 and constants maybe
        in different orderings than existing literature, but will still be in the Gell-Mann basis.
        Structure constants $f_{abc}$ is defined as

        .. math::
            [T_a, T_b] = i f_{abc} T_c

        where the group generators are $T$.

        Args:
            idxs (int): Optional postional arguments of 3 indices to return structure constant. If omitted, will return array.

        Returns:
            Union[Basic, Matrix]: If indicies are passed in, will return corresponding
            structure constant. Otherwise returns the 3dArry for entire group.
        """

        if self._structure_constants == (None, None):
            self._structure_constants = self._calculate_structure_constants()

        if len(idxs) > 0:
            if len(idxs) != 3:
                raise ValueError(
                    "3 indices need to be passed in if calling `structure_constants` with indices")
            else:
                [i, j, k] = idxs
                return self._structure_constants[0][i, j, k]
        return self._structure_constants[0]

    def _calculate_structure_constants(self):
        """Calculates the structure constants"""

        # For improved performance use explicit Symengine backend
        gens = [_CMatrix(x) for x in self.generators()]
        n = len(gens)

        f = MutableDenseNDimArray.zeros(n, n, n) * sympify("0")
        d = MutableDenseNDimArray.zeros(n, n, n) * sympify("0")

        for i in range(n):
            for j in range(n):
                for k in range(n):
                    f[i, j, k] = -2 * I * \
                        trace(commutator(gens[i], gens[j]) * gens[k])
                    d[i, j, k] = 2 * \
                        trace(commutator(
                            gens[i], gens[j], anti=True) * gens[k])
        return (f, d)

    def d_coeffecients(self, *idxs: int) -> Union[Basic, Matrix]:
        r"""Returns the d constants of the group. Indexes start at 0 and constants maybe
        in different orderings than existing literature, but will still be in the Gell-Mann basis.
        d constants $d_{abc}$ is defined as

        .. math::
            {T_a, T_b} = \frac{1}{N}\delta_{ab} + d_{abc} T_c

        where the group generators are $T$.

        Args:
            idxs (int): Optional postional arguments of 3 indices to return structure constant. If omitted, will return array.

        Returns:
            Union[Basic, Matrix]: If indicies are passed in, will return corresponding
            d constant. Otherwise returns the 3dArry for entire group.
        """
        if self._structure_constants == (None, None):
            self._structure_constants = self._calculate_structure_constants()

        if len(idxs) > 0:
            if len(idxs) != 3:
                raise ValueError(
                    "3 indices need to be passed in if calling `structure_constants` with indices")
            else:
                [i, j, k] = idxs
                return self._structure_constants[1][i, j, k]
        return self._structure_constants[1]

    def dynkin_index(self, irrep: Union[Basic, Matrix, str, int] = None, **kwargs) -> Basic:
        """Returns the dykin index for the arbitrary irreducible representation. This method 
        extends the underlying algebra method by allowing symbolic dim names to be passed.

        Args:
            irrep (Union[Basic, Matrix, str, int], optional): If None is passed, will default 
            to adjoint rep. Defaults to None.
            kwargs: Pass through to LieAlgebra.dynkin_index

        Returns:
            Basic: The irrep's dynkin index.
        """
        if irrep is None:
            return self.algebra.dynkin_index()
        elif isinstance(irrep, Matrix):
            return self.algebra.dynkin_index(irrep, **kwargs)
        elif isinstance(irrep, (Basic, str, int)):
            irrep = self.algebra.irrep_lookup(str(irrep))
            return self.algebra.dynkin_index(irrep)
        else:
            raise TypeError(
                "Only sympy basic types and sympy.Matrix are allowed.")

    def quadratic_casimir(self, irrep: Union[Basic, Matrix, str, int] = None, **kwargs) -> Basic:
        """Returns the quadratic casimir for the arbitrary irreducible representation. This method extends the underlying algebra method by allowing
        symbolic dim names to be passed.

        Args:
            irrep (Union[Basic, Matrix, str, int], optional): If None is passed, will default to adjoint rep. Defaults to None.
            kwargs: Pass through to LieAlgebra.quadratic_casimir

        Returns:
            Basic: The irrep's quadratic casimir.
        """
        if irrep is None:
            return self.algebra.quadratic_casimir()
        elif isinstance(irrep, Matrix):
            return self.algebra.quadratic_casimir(irrep, **kwargs)
        elif isinstance(irrep, (Basic, str, int)):
            irrep = self.algebra.irrep_lookup(str(irrep))
            return self.algebra.quadratic_casimir(irrep)
        else:
            raise TypeError(
                "Only sympy basic types and sympy.Matrix are allowed.")
