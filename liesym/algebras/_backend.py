import functools
from typing import Any, Callable, cast, Iterable, Tuple, TypeVar

import numpy as np
from sympy import flatten, Matrix, MatrixBase, Rational, sympify

from .. import _LieAlgebraBackend

T = TypeVar("T")


class _rust_wrapper:
    """Wraps the rust methods to and from. Formats the calls
    to rust by turning either a 2d matrix to 3d matrix of (x,y) => (x,y,[numerator,denominator])
    for preserving rational numbers. Rust returns objects as a tuple of (numerator-matrix, denominator-matrix)
    """

    def __init__(self, func: Callable[..., T]):
        functools.update_wrapper(self, func)
        self.func = func

    def __call__(self, *args, **kwargs):
        prepped_args = self._prepare_backend(args)
        result = self.func(self.instance_, *prepped_args, **kwargs)
        return self._prepare_return(result)

    def __get__(self, instance, owner):
        self.instance_ = instance
        return self.__call__

    def _prepare_return(self, ret_val: Any):
        """Prepares return type."""
        if isinstance(ret_val, tuple):
            if (
                len(ret_val) == 2
                and isinstance(ret_val[0], int)
                and isinstance(ret_val[1], int)
            ):  # Rational scalar
                return Rational(*ret_val)
            else:
                # tuple of ndarrays
                numer, denom = (x.squeeze() for x in cast(Tuple[np.ndarray], ret_val))
                shape = numer.shape

                plain_values = [
                    Rational(f"{x}/{y}")
                    for x, y in zip(numer.flatten(), denom.flatten())
                ]
                # vectorlike
                if len(shape) == 1:
                    shape = (shape[0], 1) if self.instance_.rank == 1 else (1, shape[0])

                m: Matrix = sympify(Matrix(*shape, plain_values))
                return [m.row(i) for i in range(m.shape[0])]
        elif (
            isinstance(ret_val, (int, str, float, bool)) or ret_val is None
        ):  # non-rational scalar
            return ret_val
        else:
            raise TypeError(
                f"type {type(ret_val)} is unsupported type to received from backend on function {self.func.__name__}."
            )

    def _prepare_backend(self, args: Iterable[Any]):
        """Prepares the args for sending to the backend."""
        return [self._to_rational_tuple(x) for x in args]

    def _to_rational_tuple(self, arg: Any):
        """Converts the argument into a suitable type to send to backend.
        Supported:
            - native python type
            - np.ndarray
            - sympy Rational and Matrix type
        """
        if arg is None:
            return None
        elif isinstance(arg, Rational):
            return int(arg)
        elif isinstance(arg, (int, np.ndarray)):
            return arg
        elif isinstance(arg, (list, tuple, MatrixBase)):
            mat = Matrix(arg)
            return np.array([(i.p, i.q) for i in flatten(mat)], dtype=np.int64).reshape(
                *mat.shape, 2
            )
        else:
            raise TypeError(
                f"type {type(arg)} is unsupported type to send to backend on function {self.func.__name__}."
            )


class _LieAlgebraBackendWrapped:
    @_rust_wrapper  # type:ignore[misc]
    def __init__(self, *args, **kwargs) -> None:
        # obscuring this option, used in testing
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
        algebra.simple_roots(),
        algebra.cartan_matrix.pinv(),
        algebra.omega_matrix,
        algebra.omega_matrix.pinv(),
    )
