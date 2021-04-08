import numpy as np
from sympy import Matrix, flatten, Rational

from .. import (_debug_mode, _LieAlgebraBackend)


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
    def inner(*args, **kwargs):
        cls = args[0]
        nargs = [_to_rational_tuple(x) for x in args[1:]]
        result = func(cls, *nargs, **kwargs)
        numer, denom = (x.squeeze() for x in result)
        shape = numer.shape
        plain_values = [Rational(f"{x}/{y}")
                        for x, y in zip(numer.flatten(), denom.flatten())]
        m = Matrix(*shape, plain_values)
        return [m.row(i) for i in range(m.shape[0])]
    return inner


class _LieAlgebraBackendWrapped:
    @_rust_new
    def __init__(self, *args, **kwargs):
        # obscuring this option
        backend = kwargs.get("backend", _LieAlgebraBackend)
        self.backend = backend(*args)

    @_rust_wrapper
    def orbit(self, weight, stabilizers):
        return self.backend.orbit(weight, stabilizers)

    @_rust_wrapper
    def root_system(self):
        return self.backend.root_system()


def create_backend(algebra):
    return _LieAlgebraBackendWrapped(
        algebra.rank,
        algebra.roots,
        algebra.simple_roots,
        algebra.cartan_matrix,
        algebra.cartan_matrix.pinv(),
        algebra.omega_matrix,
        algebra.omega_matrix.pinv(),
        algebra.cocartan_matrix,
    )
