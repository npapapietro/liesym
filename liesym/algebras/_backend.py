import numpy as np
from sympy import Matrix, flatten, Rational

from .. import (_debug_mode, _LieAlgebraBackend)


def _annotate(M: Matrix, basis: str) -> Matrix:
    """Adds basis attribute to sympy.Matrix"""
    setattr(M, "basis", basis)
    return M


def _to_rational_tuple(obj):
    if obj is None:
        return
    elif isinstance(obj, list) or isinstance(obj, tuple):
        obj = Matrix(obj)
    elif isinstance(obj, Rational):
        return int(obj)
    elif isinstance(obj, int) or isinstance(obj, np.ndarray):
        return obj
    else:
        pass
    x = flatten(obj)
    return np.array([(i.p, i.q) for i in x], dtype=np.int64).reshape(*obj.shape, 2)


def _rust_new(func):
    def inner(*args, **kwargs):
        cls = args[0]
        nargs = [_to_rational_tuple(x) for x in args[1:]]
        return func(cls, *nargs, **kwargs)
    return inner


def _rust_wrapper(func):
    """Wraps the rust methods to and from. Formats the calls
    to rust by turning either a 2d matrix to 3d matrix of (x,y) => (x,y,[numerator,denominator])
    for preserving rational numbers. Rust returns objects as a tuple of (numerator-matrix, denominator-matrix)
    """
    def inner(*args, **kwargs):
        cls = args[0]
        rank = cls.rank

        nargs = [_to_rational_tuple(x) for x in args[1:]]
        result = func(cls, *nargs, **kwargs)

        if isinstance(result, (int, str, float, bool)) or result is None:
            return result

        if not isinstance(result, tuple):
            return

        # Tuple of ints
        if isinstance(result[0], int) and isinstance(result[1], int):
            return Rational(*result)

        # tuple of ndarrays
        numer, denom = (x.squeeze() for x in result)
        shape = numer.shape

        plain_values = [Rational(f"{x}/{y}")
                        for x, y in zip(numer.flatten(), denom.flatten())]
        # vectorlike
        if len(shape) == 1:
            shape = (shape[0], 1) if rank == 1 else (1, shape[0])

        m = Matrix(*shape, plain_values)
        return [m.row(i) for i in range(m.shape[0])]
    return inner


class _LieAlgebraBackendWrapped:
    @_rust_new
    def __init__(self, *args, **kwargs):
        # obscuring this option
        backend = kwargs.get("backend", _LieAlgebraBackend)
        self.rank = args[0]
        self.backend = backend(*args)

    @_rust_wrapper
    def orbit(self, weight, stabilizers):
        return self.backend.orbit(weight, stabilizers)

    @_rust_wrapper
    def root_system(self):
        return self.backend.root_system()

    @_rust_wrapper
    def tensor_product_decomposition(self, irrepA, irrepB):
        return self.backend.tensor_product_decomposition(irrepA, irrepB)

    @_rust_wrapper
    def dim(self, irrep):
        return self.backend.dim(irrep)

    @_rust_wrapper
    def get_irrep_by_dim(self, dim, max_dd):
        return self.backend.irrep_by_dim(dim, max_dd)

    @_rust_wrapper
    def index_irrep(self, irrep, dim):
        return self.backend.index_irrep(irrep, dim)

    @_rust_wrapper
    def conjugate(self, irrep):
        return self.backend.conjugate_irrep(irrep)


def create_backend(algebra):
    return _LieAlgebraBackendWrapped(
        algebra.rank,
        algebra.n_pos_roots,
        algebra.simple_roots,
        algebra.cartan_matrix,
        algebra.cartan_matrix.pinv(),
        algebra.omega_matrix,
        algebra.omega_matrix.pinv(),
        algebra.cocartan_matrix,
    )
