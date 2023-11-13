from __future__ import annotations

from typing import (
    Any,
    Callable,
    cast,
    List,
    Literal,
    Optional,
    overload,
    Tuple,
    TYPE_CHECKING,
    TypeVar,
    Union,
)

if TYPE_CHECKING:
    F = TypeVar("F", bound=Callable)

    def lru_cache(f: F) -> F:
        ...

else:
    from functools import lru_cache

from sympy import Basic, I, Matrix, Symbol, sympify, trace
from sympy.core.backend import Matrix as _CMatrix
from sympy.matrices import MatrixBase
from sympy.tensor.array.dense_ndim_array import MutableDenseNDimArray

from ..algebras import BASIS, LieAlgebra, NumericSymbol


def _cast_func(typ):
    def inner(f: Callable[..., Any]):
        return f

    return cast(Callable[..., Callable[..., typ]], inner)


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
        return A * B + B * A  # type: ignore[operator]
    return A * B - B * A  # type: ignore[operator]


def _not_impl():
    raise NotImplementedError("Call this method on a concrete subclass")


class Group(Basic):
    """The base class for all (lie) groups. The methods and properties
    in this class are basis independent and apply in a general sense. In
    order to write down the roots as matricies and vectors, we choose a
    representation.
    """

    _group: str

    def __new__(cls, group: str, dim: int):
        obj = super().__new__(cls, sympify(dim))
        obj._group = group
        return obj

    @property
    def group(self) -> str:
        """Group type"""
        return self._group

    @property
    def dimension(self) -> Basic:
        """Group dimension as sympy object"""
        return self.args[0]

    @_cast_func(List[Any])
    def generators(self, *args, **kwargs):
        """Generalized matrix generators of the group.

        Abstract
        """
        raise NotImplementedError("Call this method on a concrete subclass")

    @_cast_func(List[Any])
    def product(self, *args, **kwargs):
        """Calculates the product between several group representations.
        If backing algebra is a Lie Algebra, defaults to `LieAlgebra.tensor_product_decomposition`.

        Abstract
        """
        raise NotImplementedError("Call this method on a concrete subclass")

    def conjugate(self, rep, symbolic=False):
        """Returns the conjugate representation. If the incoming rep is symbolic/named then it will return as such.

        Abstract
        """

    def irrep_lookup(self, irrep):
        """Returns the mathematical representation to the common name. Example would be returning `3` in SU(3) as `Matrix([[1,0]])`

        Abstract
        """


class LieGroup(Group):
    """Group that has a Lie Algebra associated with it."""

    def __init__(self, *args, **kwargs):
        """Used to set lazy properties"""
        self._algebra = None
        self._structure_constants = (None, None)

    @property
    def algebra(self) -> LieAlgebra:
        """Backing Lie Algebra"""
        return self._algebra

    @overload
    def product(
        self, *args: Matrix, basis: BASIS = "omega", return_type: None = None
    ) -> List[Matrix]:
        ...

    @overload
    def product(
        self,
        *args: Union[NumericSymbol, str],
        basis: BASIS = "omega",
        return_type: None = None,
    ) -> List[NumericSymbol]:
        ...

    @overload
    def product(
        self,
        *args: Union[Matrix, NumericSymbol, str],
        basis: BASIS = "omega",
        return_type: Literal["matrix"],
    ) -> List[Matrix]:
        ...

    @overload
    def product(
        self,
        *args: Union[Matrix, NumericSymbol, str],
        basis: BASIS = "omega",
        return_type: Literal["rep"],
    ) -> List[NumericSymbol]:
        ...

    def product(
        self,
        *args: Union[Matrix, NumericSymbol, str],
        basis: BASIS = "omega",
        return_type: Optional[Literal["matrix", "rep"]] = None,
    ) -> Union[List[Matrix], List[NumericSymbol]]:
        """Uses tensor product decomposition to find the products between the
        representations. Supported kwargs can be found on `LieAlgebra.tensor_product_decomposition`

        Args:
            args: Objects to take product of
            basis ("ortho" | "omega" | "alpha", optional): Basis of incoming weights and result. If not set, will implicitly set. Defaults to 'omega'.
            return_type ("matrix" | "rep" | None, optional): Returns either as matrices or symbolic representation. Will default to the type of the args unless explicitly set here. Defaults to None.

        Returns:
            Union[List[Matrix], List[NumericSymbol]]: Tensor sum of the product.

        Examples
        ========
        >>> from liesym import SO
        >>> from sympy import Matrix
        >>> so10 = SO(10)
        >>> so10.product(Matrix([[1,0,0,0,0]]),Matrix([[1,0,0,0,0]]))
        [Matrix([[0, 0, 0, 0, 0]]), Matrix([[0, 1, 0, 0, 0]]), Matrix([[2, 0, 0, 0, 0]])]
        """

        # check args types are all the same
        if not all([type(x) for x in args]):
            raise TypeError(
                "All product types must be the same type, no mixing allowed."
            )

        if return_type not in ["matrix", "rep", None]:
            raise TypeError("Allowed values of return_type are 'matrix', 'rep' or None")

        ret_type = None

        if all([isinstance(x, MatrixBase) for x in args]):
            results = self.algebra.tensor_product_decomposition(
                *cast(Tuple[Matrix, ...], args), basis=basis
            )
            ret_type = return_type or "matrix"
        if all([isinstance(x, str) or isinstance(x, Symbol) for x in args]):
            mats = [self.algebra.irrep_lookup(x) for x in args]
            results = self.algebra.tensor_product_decomposition(
                *cast(Tuple[Matrix, ...], mats), basis=basis
            )
            ret_type = return_type or "rep"
        else:
            raise TypeError(
                "Unsupported product arg. All product types must be either a matrix type or str/symbol type."
            )

        if ret_type == "matrix":
            return results
        else:
            return [self.algebra.dim_name(x) for x in results]

    def conjugate(self, rep, symbolic=False):
        r"""Uses the underlying algebra to find the conjugate representation.

        Examples
        ========
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
                    "3 indices need to be passed in if calling `structure_constants` with indices"
                )
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
                    f[i, j, k] = -2 * I * trace(commutator(gens[i], gens[j]) * gens[k])
                    d[i, j, k] = 2 * trace(
                        commutator(gens[i], gens[j], anti=True) * gens[k]
                    )
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
                    "3 indices need to be passed in if calling `structure_constants` with indices"
                )
            else:
                [i, j, k] = idxs
                return self._structure_constants[1][i, j, k]
        return self._structure_constants[1]

    def dynkin_index(
        self, irrep: Optional[Union[Basic, Matrix, str, int]] = None, **kwargs
    ) -> Basic:
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
            raise TypeError("Only sympy basic types and sympy.Matrix are allowed.")

    def quadratic_casimir(
        self,
        irrep: Optional[Union[Basic, Matrix, str, int]] = None,
        basis: BASIS = "omega",
    ) -> Basic:
        """Returns the quadratic casimir for the arbitrary irreducible representation. This method extends the underlying algebra method by allowing
        symbolic dim names to be passed.

        Args:
            irrep (Union[Basic, Matrix, str, int], optional): If None is passed, will default to adjoint rep. Defaults to None.
            kwargs: Pass through to LieAlgebra.quadratic_casimir

        Returns:
            Basic: The irrep's quadratic casimir.
        """
        if irrep is None:
            return self.algebra.quadratic_casimir(basis=basis)
        elif isinstance(irrep, Matrix):
            return self.algebra.quadratic_casimir(irrep, basis=basis)
        elif isinstance(irrep, (Basic, str, int)):
            irrep = self.algebra.irrep_lookup(irrep)
            return self.algebra.quadratic_casimir(irrep, basis=basis)
        else:
            raise TypeError(
                "Only sympy basic types, str, int and sympy.Matrix are allowed."
            )

    @_cast_func(List[Any])
    @lru_cache
    def generators(self, cartan_only: bool = False, **kwargs: bool):
        """Returns the generators representations of the group.

        Args:
            cartan_only (bool, optional): Only return the cartan generators (diagonalized generators). Defaults to False.

        Abstract
        """
        generators = self._calc_generator(**kwargs)

        if cartan_only:
            return [x for x in generators if x.is_diagonal()]
        else:
            return generators

    def _calc_generator(self, **kwargs):
        raise NotImplementedError("This method needs calculation")
