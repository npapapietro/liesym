import numpy as np
from sympy import Matrix, flatten, Rational, re, im

from .. import _structure_constants_backend

def _decompose_complex_rationals(M: Matrix):
    x = flatten(M)

    reals = []
    imags = []

    for i in x:
        real = Rational(re(i))
        imag = Rational(im(i))
        reals.append((real.p, real.q))
        imags.append((imag.p, imag.q))
    return reals, imags    
    

def _structure_constants(generators: list):

    N = len(generators)
    shape = generators[0].shape

    reals = []
    imags = []
    for i in generators:
        real, imag = _decompose_complex_rationals(i)
        reals.append(real)
        imags.append(imag)
    
    reals = np.array(reals, dtype=np.int64).reshape(N, *shape, 2)
    imags = np.array(imags, dtype=np.int64).reshape(N, *shape, 2)

    result = _structure_constants_backend(reals, imags)
