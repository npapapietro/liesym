import numpy as np
from sympy import Matrix, flatten, Rational

from .. import (_debug_mode, _unstabilized_orbit,
                _stabilized_orbit, _tensordecomposition)

def _prepare_matrix(obj):
    if isinstance(obj, list) or isinstance(obj, tuple):
        obj = Matrix(obj)
    elif isinstance(obj, int) or isinstance(obj, float) or isinstance(obj, np.ndarray):
        return obj
    x = flatten(obj)
    return np.array([(i.p, i.q) for i in x], dtype=np.int64).reshape(*obj.shape, 2)

def _rust_wrapper(func):
    def inner(*args, **kwargs):
        args = [_prepare_matrix(x) for x in args]
        result = func(*args, **kwargs)
        numer, denom = (x.squeeze() for x in result)
        shape = numer.shape
        plain_values = [Rational(f"{x}/{y}")
                        for x, y in zip(numer.flatten(), denom.flatten())]
        m = Matrix(*shape, plain_values)
        return [m.row(i) for i in range(m.shape[0])]
    return inner

@_rust_wrapper
def _orbit(*args, **kwargs):
    try:
        return _unstabilized_orbit(*args)
    except Exception as e:
        raise

@_prepare_matrix
def _orbit_stab(*args, **kwargs):
    try:
        return _stabilized_orbit(*args)
    except Exception as e:
        raise

def orbit(algebra, weight, stabilizers=None, **kwargs):
    if isinstance(kwargs.get("debug", False), bool):
        _debug_mode(kwargs.get("debug", False))

    args = [
        algebra.simple_roots,
        algebra.omega_matrix.pinv(),
        algebra.omega_matrix,
        algebra.cartan_matrix.pinv(),
        algebra.cocartan_matrix.T,
        algebra.roots // 2,
        algebra.rank,
        weight
    ]

    if stabilizers is None:
        return _orbit(*args)

    if not isinstance(stabilizers, list):
        raise ValueError("Stabilizers must be list of integers.")

    stabilizers = np.array(stabilizers, dtype=np.int64)
    args += stabilizers

    return _orbit_stab(*args)