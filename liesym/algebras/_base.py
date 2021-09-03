from __future__ import annotations

from sympy.core import Basic
from sympy import Matrix, Symbol, sympify
from typing import Tuple, Union
from copy import deepcopy
from functools import cmp_to_key
import re

from ._methods import (
    _cartan_matrix,
    _cocartan_matrix,
    _quadratic_form,
    _reflection_matricies,
    _annotate_matrix,
    Basis,
    _basis_lookup
)

from ._backend import create_backend


class NumericSymbol(Symbol):
    """Extension of Sympy symbol that allows
    latex formatting but also tracks the underlying 
    integer value. Useful for dimension representations
    of irreps"""
    def __new__(cls, dim: int, fmtted_dim: str):
        obj = super().__new__(cls, fmtted_dim)
        obj.numeric_dim = int(dim)
        return obj

    @classmethod
    def from_symbol(cls, symbol: Symbol):
        """Converts from sympy.Symbol into NumericSymbol by
        regex search for digits from latex display pattern and 
        returns a NumericSymbol. Will raise if no numeric is 
        present in the symbol.
        """
        try:
            s = symbol.__str__()
            num = re.findall(r"\d+", s)[0]
            return cls(int(num), s)
        except (IndexError, ValueError):
            raise ValueError("Could not extract numerical from sympy.Symbol")


class LieAlgebra(Basic):
    """The base class for all lie algebras. The methods and properties
    in this class are basis independent and apply in a general sense. In
    order to write down the roots as matricies and vectors, we choose a 
    representation.

    """
    def __new__(cls, series: str, rank: int):
        """
        Returns a new instance of a Sympy object

        Args:
            series (str): The series type
            rank (int): The rank of the algebra
        """
        obj = super().__new__(cls, sympify(rank))
        obj._series = series
        return obj

    def __init__(self, *args, **kwargs):
        """Used to set lazy properties
        """
        self._simple_roots = None
        self._positive_roots = None
        self._cartan_matrix = None
        self._omega_matrix = None
        self._quadratic_form = None
        self._cocartan_matrix = None
        self._reflection_matricies = None
        self._fundamental_weights = None
        self._backend = None
        self._root_system = None

    @property
    def series(self) -> str:
        """Algebra series type
        """
        return self._series

    @property
    def rank(self) -> int:
        """Algebra rank
        """
        return self.args[0]

    @property
    def dimension(self) -> int:
        """Algebra dimension

        Abstract
        """

    @property
    def n_pos_roots(self) -> int:
        """Total number of positive roots in the algebra

        Abstract
        """
    @property
    def n_roots(self) -> int:
        """Total number of roots in the algebra"""
        return 2 * self.n_pos_roots + self.rank

    @property
    def simple_roots(self) -> list[Matrix]:
        """Returns a list of Sympy matrix (1,dimension)
        objects representing a chosen basis of the algebra.

        Basis: Orthogonal


        This method can be overridden to choose your own basis,
        be sure to do this before any other properties are called
        as they are lazily evaluated and the simple_roots define
        the entire representation of the algebra.

        Examples
        ========

        .. code-block:: python

            from liesym import F4

            algebra = F4()
            my_simple_roots = [
                    # my basis
                ]
            algebra.simple_roots = my_simple_roots

        """
        return [_annotate_matrix(x) for x in self._simple_roots]

    @simple_roots.setter
    def simple_roots(self, val: list[Matrix]):
        """Overrides the default representation of the algebras simple_roots.
        Please ensure that roots are in Orthogonal Basis
        """
        assert len(val) == len(
            self._simple_roots), "Incorrect number of simple roots"
        self._simple_roots = val

    @property
    def cartan_matrix(self) -> Matrix:
        r"""For a given simple Lie algebra the elements $a_{ij}$ can be
        generated by

        .. math::
            a_{ji} = 2 \langle\alpha_i, \alpha_j\rangle / \langle\alpha_j, \alpha_j\rangle

        where $a_i$ is the i'th simple root and $\langle,\rangle$ is the scalar product.

        Sources:
            - https://en.wikipedia.org/wiki/Cartan_matrix
            - https://mathworld.wolfram.com/CartanMatrix.html

        Returns:
            Matrix: Cartan Matrix as a Sympy object
        """
        if self._cartan_matrix is None:
            self._cartan_matrix = _cartan_matrix(self.simple_roots)
        return self._cartan_matrix

    @property
    def cocartan_matrix(self) -> Matrix:
        """The cocartan matrix rows are generated from the coroots of 
        the algebra such that multiplication by a simple root will
        generate a row of the cartan matrix.

        Returns:
            Matrix: Cocartan Matrix as a Sympy object
        """
        if self._cocartan_matrix is None:
            self._cocartan_matrix = _cocartan_matrix(self.simple_roots)
        return self._cocartan_matrix

    @property
    def omega_matrix(self) -> Matrix:
        """The rows of the omega matrix are the fundamental weights
        of the algebra.

        Returns:
            Matrix: Omega Matrix as a Sympy object
        """
        if self._omega_matrix is None:
            self._omega_matrix = self.cocartan_matrix.pinv().T
        return self._omega_matrix

    @property
    def metric_tensor(self) -> Matrix:
        """Also known as the quadratic form, the metric tensor
        serves as the metrix for the inner product of two roots or weights
        when they are not in the orthogonal basis.

        Returns:
            Matrix: Metric Tensor as a Sympy object
        """
        if self._quadratic_form is None:
            self._quadratic_form = _quadratic_form(
                self.cartan_matrix, self.simple_roots)
        return self._quadratic_form

    @property
    def reflection_matricies(self) -> list[Matrix]:
        """Returns a list of reflection matrices built from
        rotations about each simple root.

        Returns:
            list[Matrix]: list of Sympy Matrices
        """
        if self._reflection_matricies is None:
            self._reflection_matricies = _reflection_matricies(
                self.simple_roots)
        return self._reflection_matricies

    @property
    def fundamental_weights(self) -> list[Matrix]:
        """Returns the fundamental weights of the algebra. 

        Basis: Orthogonal

        Returns:
            list[Matrix]: list of Sympy Matrices
        """
        if self._fundamental_weights is None:
            self._fundamental_weights = [
                _annotate_matrix(self.omega_matrix.row(i))
                for i in range(self.omega_matrix.rows)
            ]
        return self._fundamental_weights

    @property
    def positive_roots(self) -> list[Matrix]:
        """Returns the postive roots of the algebra. They are sorted 
        first by their distance from the highest root and then by 
        tuple ordering (convention).

        Basis: Orthogonal

        Returns:
            list[Matrix]: list of Sympy Matrices
        """
        if self._positive_roots is None:
            self._positive_roots = self.root_system()[:self.n_pos_roots]
        return self._positive_roots

    @property
    def _backend_instance(self):
        if self._backend is None:
            self._backend = create_backend(self)
        return self._backend

    def orbit(self, weight: Matrix, stabilizers=None, basis="ortho") -> list[Matrix]:
        """Returns the orbit of the weight or root by reflecting it
        a plane. A stabilizer may be passed to calculate the orbit using
        the Orbit-Stabilizer theorem.

        Basis: Ortho

        Args:
            weight (Matrix): A Matrix of shape (1, rank)
            stabilizer (Iterable of ints, optional): Per Orbit-Stabilizer
            theorem, integer iterable of simple root indexes. Defaults to None.

        Sources:
        - https://en.wikipedia.org/wiki/Coadjoint_representation#Coadjoint_orbit
        - https://en.wikipedia.org/wiki/Group_action#Orbits_and_stabilizers

        """

        weight = self.to_ortho(weight, "ortho")
        return [self.to_ortho(x, "ortho") for x in self._backend_instance.orbit(weight, stabilizers)]

    def dim_name(self, irrep: Matrix, basis="omega") -> NumericSymbol:
        r"""Returns a sympy formatted symbol for the irrep.
        This is commonly used in physics literature. Returns
        a NumericSymbol object that is a simple extension of 
        sympy.Symbol.      

        Examples
        =========
        >>> from liesym import A
        >>> from sympy import Matrix
        >>> a3 = A(3)
        >>> assert str(a3.dim_name(Matrix([[1, 1, 0]]))) == '\\bar{20}'           
        """
        irrep = self.to_omega(irrep, basis)

        dim = self.dim(irrep)
        max_dd = self.max_dynkin_digit(irrep)
        same_dim_irreps: list[Matrix] = self.get_irrep_by_dim(dim, max_dd)
        num_primes = 0
        conjugate = 0
        so8label = ""

        if len(same_dim_irreps) > 1:
            # group by index
            index_pairs = {}  # type: ignore
            for i in same_dim_irreps:
                index = self._backend_instance.index_irrep(i, dim)
                index_pairs[index] = index_pairs.get(index, []) + [i]

            groups = [sorted(dimindex, key=cmp_to_key(self._dimindexsort))
                      for dimindex in index_pairs.values()]
            positions = []
            for id1, grps in enumerate(groups):
                for id2, g in enumerate(grps):
                    if g == irrep:
                        positions.append([id1, id2])
            [num_primes, conjugate] = positions[0]
            so8label = self._is_s08(irrep)
        has_conjugate = conjugate == 1 if so8label == "" else False
        return self._dim_name_fmt(dim, has_conjugate, num_primes, so8label)

    def irrep_lookup(self, dim: Union[Symbol, str]) -> Matrix:
        """Returns the irrep matrix for the dimension.

        Args:
            dim (Union[Symbol, str]): Can either be a sympy.Symbol or string.

        Raises:
            KeyError: Dim not found

        Returns:
            Matrix: Returns irrep in Omega basis


        Examples
        ========
        >>> from liesym import A
        >>> A3 = A(3)
        >>> A3.irrep_lookup(r"\\bar{4}")
        Matrix([[0, 0, 1]])
        >>> A3.irrep_lookup("4")
        Matrix([[1, 0, 0]])
        """
        if isinstance(dim, str):
            dim = Symbol(dim)

        if isinstance(dim, Symbol) and not isinstance(dim, NumericSymbol):
            dim = NumericSymbol.from_symbol(dim)

        n_dim = dim.numeric_dim

        max_dynkin_digit = 3
        dd = 0
        while dd < max_dynkin_digit:
            dd += 1
            for c in self.get_irrep_by_dim(n_dim, dd):
                if self.dim_name(c) == dim:
                    return c
        raise KeyError(f"Irrep {dim} not found.")

    def conjugate(self, irrep: Matrix) -> Matrix:
        """Finds the conjugate irrep. If it is the same
        as the original irrep, you have a Real Irrep, otherwise
        it's a Complex Irrep.

        Examples
        ========
        .. code-block:: python

            from liesym import A,D
            from sympy import Matrix

            SU4 = A(3)
            irrep_20 = Matrix([[0,1,1]])
            irrep_20bar = Matrix([[1,1,0]])
            assert irrep_20 == SU4.conjugate(irrep_20bar)

            SO10 = D(5)
            irrep_10 = Matrix([[1, 0, 0, 0, 0]])
            assert irrep_10 == SO10.conjugate(irrep_10)
        """
        return self.to_omega(self._backend_instance.conjugate(irrep)[0], "omega")

    def _is_s08(self, irrep):
        return ""

    def _dimindexsort(self, irrep1, irrep2):
        cong1 = self._congruency_class(irrep1)
        cong2 = self._congruency_class(irrep2)
        if isinstance(cong1, tuple):
            return 1 if cong1[-1] <= cong2[-1] else -1
        else:
            return -1 if cong1 <= cong2 else 1

    def _congruency_class(self, irrep):
        return 0

    def max_dynkin_digit(self, irrep: Matrix) -> int:
        """Returns the max Dynkin Digit for the representations"""
        pass

    def _dim_name_fmt(self, dim: int, conj=False, primes=0, sub="") -> NumericSymbol:
        if conj:
            irrep = r"\bar{" + str(dim) + "}"
        else:
            irrep = str(dim)

        if primes > 0:
            irrep += r"^{" + " ".join([r"\prime"] * primes) + r"}"

        if sub != "":
            irrep += r"_{" + str(sub) + r"}"

        return NumericSymbol(dim, irrep)

    def get_irrep_by_dim(self, dim: int, max_dd: int = 3, with_symbols=False) -> list[Union[Matrix, Tuple[Matrix, NumericSymbol]]]:
        r"""Gets all irreps by dimension and max dynkin digit. `max_dd` is . This algorithm brute forces searches by using `itertools.product`
        which can become expensive for large so searching max_dd > 3 will be 
        very expensive

        Args:
            dim (int): Dimension to query
            max_dd (int, optional): The max dynkin digit to use. Defaults to 3.
            with_symbols (bool, optional): Returns list of tuples of rep and latex fmt. Defaults to False.

        Returns:
            list[Union[Matrix, Tuple[Matrix,NumericSymbol]]]: If `with_symbols=True` will return a list of tuples.


        Examples
        =========
        >>> from liesym import A
        >>> from sympy import Matrix
        >>> a3 = A(3)
        >>> expected = a3.get_irrep_by_dim(20)
        >>> result = [
        ... Matrix([[1, 1, 0]]),
        ... Matrix([[0, 1, 1]]),
        ... Matrix([[0, 2, 0]]),
        ... Matrix([[3, 0, 0]]),
        ... Matrix([[0, 0, 3]])]
        >>> assert expected == result
        >>> a3.get_irrep_by_dim(20, with_symbols=True)
        [(Matrix([[1, 1, 0]]), \bar{20}), (Matrix([[0, 1, 1]]), 20), (Matrix([[0, 2, 0]]), 20^{\prime}), (Matrix([[3, 0, 0]]), \bar{20}^{\prime \prime}), (Matrix([[0, 0, 3]]), 20^{\prime \prime})]
        """
        backend_results: list[Matrix] = self._backend_instance.get_irrep_by_dim(
            dim, max_dd)
        results = [self.to_omega(x, "omega") for x in backend_results]
        if with_symbols:
            results = [(x, self.dim_name(x)) for x in results]
        return results

    def dim(self, irrep: Matrix, basis="omega") -> int:
        r"""Returns the dimension of the weight, root or irreducible representations.
        This follows Weyl's dimension formula:

        .. math::
            dim(w) = \prod_{\alpha\in\Delta^{+}} \frac{\langle \alpha, w + \rho\rangle}{\langle\alpha,\rho\rangle}

        where $\Delta^{+}$ are the positive roots and $rho$ is the sum of
        the positive roots: `[1] * rank`.

        Examples
        ========
        >>> from liesym import A
        >>> from sympy import Matrix
        >>> a2 = A(2)
        >>> assert a2.dim(Matrix([[1,0]])) == 3
        """
        basis = _basis_lookup(basis)
        _annotate_matrix(irrep, basis)
        irrep = self.to_omega(irrep)

        return self._backend_instance.dim(irrep)

    def root_system(self) -> list[Matrix]:
        """Returns the entire rootsystem of the algebra. This
        includes the positive, negative and zeros of the algebra.

        Basis: Orthogonal

        Returns:
            list[Matrix]: list of ordered roots.
        """
        if self._root_system is None:
            self._root_system = [_annotate_matrix(
                x) for x in self._backend_instance.root_system()]
        return self._root_system

    def tensor_product_decomposition(self, weights: list[Matrix], basis="omega", **_) -> list[Matrix]:
        """Returns the tensor product between irreducible representations
        as a the tensor sum of the irreducible representations of their
        highest weights. This algorithm is based on Klimky's formula.


        Args:
            weights (list[Matrix]): A list of fundamental weights to take the tensor product between
            basis (str, Optional): Basis of incoming weights. If not set, will implicitly set. Defaults to 'omega'.

        Returns:
            list[Matrix]: list of weights decomposed from the tensor product. Basis: Omega


        Examples
        =========
        >>> from liesym import A
        >>> from sympy import Matrix
        >>> a2 = A(2)
        >>> results = a2.tensor_product_decomposition([Matrix([[1,0]]), Matrix([[1,0]])])
        >>> print(results)
        [Matrix([[0, 1]]), Matrix([[2, 0]])]


        """
        weights = [self.to_omega(x, basis) for x in weights]
        w = deepcopy(weights)
        i = w.pop()
        j = w.pop()

        decomp = self._backend_instance.tensor_product_decomposition(i, j)

        while len(w) > 0:
            j = w.pop()
            results = []
            for i in decomp:
                # i,j reversed because pop takes from -1 index
                results += self._backend_instance.tensor_product_decomposition(
                    j, i)
            decomp = results
        return [self.to_omega(x, "omega") for x in decomp]

    def to_ortho(self, x: Matrix, basis=None) -> Matrix:
        """Rotates to orthogonal basis

        Args:
            x (Matrix): Matrix to be rotated
            basis (optional): If `basis` attribute is not set on `x` define it here. Defaults to None.

        Raises:
            ValueError: If no `x.basis` is set and None is passed to `basis` kwarg. 

        Returns:
            Matrix: Matrix in orthogonal basis.
        """
        basis = _basis_lookup(basis)
        _annotate_matrix(x, basis)

        if x.basis is Basis.ORTHO:
            r = x
        elif x.basis is Basis.OMEGA:
            r = x * self.omega_matrix
        elif x.basis is Basis.ALPHA:
            r = x * self.cartan_matrix * self.omega_matrix
        else:
            raise ValueError(
                "Basis arg cannot be None if attribute `basis` has not been set on Matrix.")

        r.basis = Basis.ORTHO
        return r

    def to_omega(self, x: Matrix, basis=None) -> Matrix:
        """Rotates to omega basis

        Args:
            x (Matrix): Matrix to be rotated
            basis (optional): If `basis` attribute is not set on `x` define it here. Defaults to None.

        Raises:
            ValueError: If no `x.basis` is set and None is passed to `basis` kwarg. 

        Returns:
            Matrix: Matrix in omega basis.
        """
        basis = _basis_lookup(basis)
        _annotate_matrix(x, basis)

        if x.basis is Basis.OMEGA:
            r = x
        elif x.basis is Basis.ORTHO:
            r = x * self.omega_matrix.pinv()
        elif x.basis is Basis.ALPHA:
            r = x * self.cartan_matrix
        else:
            raise ValueError(
                "Basis arg cannot be None if attribute `basis` has not been set on Matrix.")

        r.basis = Basis.OMEGA
        return r

    def to_alpha(self, x: Matrix, basis=None) -> Matrix:
        """Rotates to alpha basis

        Args:
            x (Matrix): Matrix to be rotated
            basis (optional): If `basis` attribute is not set on `x` define it here. Defaults to None.

        Raises:
            ValueError: If no `x.basis` is set and None is passed to `basis` kwarg. 

        Returns:
            Matrix: Matrix in alpha basis.
        """
        basis = _basis_lookup(basis)
        _annotate_matrix(x, basis)

        if x.basis is Basis.ALPHA:
            r = x
        elif x.basis is Basis.ORTHO:
            r = x * self.omega_matrix.pinv() * self.cartan_matrix.pinv()
        elif x.basis is Basis.OMEGA:
            r = x * self.cartan_matrix.pinv()
        else:
            raise ValueError(
                "Basis arg cannot be None if attribute `basis` has not been set on Matrix.")

        r.basis = Basis.ALPHA
        return r
