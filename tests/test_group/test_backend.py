from liesym.groups._backend import _decompose_complex_rationals, _structure_constants

from sympy import Matrix, I


def test_complex_xform():

    M = Matrix([
        [I, 1],
        [-1, I + 2]
    ])

    real, imag = _decompose_complex_rationals(M)

    expected_real = [
        (0, 1), (1, 1), (-1, 1), (2, 1)
    ]

    expected_imag = [
        (1, 1), (0, 1), (0, 1), (1, 1)
    ]

    assert expected_real == real
    assert expected_imag == imag


def test_structure_constants():

    pauli = [
        Matrix([
            [0, 1],
            [1, 0]]) / 2,
        Matrix([
            [0, -I],
            [I,  0]]) / 2,
        Matrix([
            [1,  0],
            [0, -1]]) / 2
    ]

    _structure_constants(pauli)