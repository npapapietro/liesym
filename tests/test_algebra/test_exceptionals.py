from liesym import F4, E, Basis
from sympy import Matrix, Rational, S


def test_F4():
    F4_ = F4()

    # test subclass items
    assert F4_.dimension == 4
    assert F4_.n_pos_roots == 24

    assert F4_.simple_roots == [
        Matrix([[1, -1, 0, 0]]),
        Matrix([[0, 1, -1, 0]]),
        Matrix([[0, 0, 1, 0]]),
        Matrix([[-S.Half, -S.Half, -S.Half, -S.Half]]),
    ]

    fw = F4_.fundamental_weights[0]
    assert fw.basis == Basis.ORTHO
    assert F4_.to_omega(fw) == Matrix([[1, 0, 0, 0]])

    # baseclass generated
    assert F4_.cartan_matrix == Matrix(
        [[2, -1, 0, 0], [-1, 2, -2, 0], [0, -1, 2, -1], [0, 0, -1, 2]])
    assert F4_.cocartan_matrix == Matrix(
        [[1, -1, 0, 0], [0, 1, -1, 0], [0, 0, 2, 0], [-1, -1, -1, -1]])
    assert F4_.omega_matrix == Matrix(
        [[1, 0, 0, -1], [1, 1, 0, -2], [S.Half, S.Half, S.Half, -3*S.Half], [0, 0, 0, -1]])
    assert F4_.metric_tensor == Matrix(
        [[2, 3, 2, 1], [3, 6, 4, 2], [2, 4, 3, 3*S.Half], [1, 2, 3*S.Half, 1]])
    assert F4_.reflection_matricies == [
        Matrix([
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]),
        Matrix([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]]),
        Matrix([
            [1,  0,  0,  0],
            [0,  1,  0,  0],
            [0,  0, -1,  0],
            [0,  0,  0,  1]]),
        Matrix([
            [S.Half,  -S.Half,  -S.Half,  -S.Half],
            [-S.Half,  S.Half, -S.Half, -S.Half],
            [-S.Half, -S.Half,  S.Half, -S.Half],
            [-S.Half, -S.Half, -S.Half,  S.Half]])]
    assert F4_.fundamental_weights == [Matrix([[1, 0, 0, -1]]), Matrix(
        [[1, 1, 0, -2]]), Matrix([[S.Half, S.Half, S.Half, -3*S.Half]]), Matrix([[0, 0, 0, -1]])]

    # backend
    assert F4_.root_system() == [
        Matrix([[1, 0, 0, 0]]),
        Matrix([[-1, 1, 0, 0]]),
        Matrix([[0, -1, 2, 0]]),
        Matrix([[0, 0, 0, 1]]),
        Matrix([[0, 0, 1, -1]]),
        Matrix([[0, 1, -2, 2]]),
        Matrix([[0, 1, -1, 0]]),
        Matrix([[1, -1, 0, 2]]),
        Matrix([[-1, 0, 0, 2]]),
        Matrix([[0, 1, 0, -2]]),
        Matrix([[1, -1, 1, 0]]),
        Matrix([[-1, 0, 1, 0]]),
        Matrix([[1, -1, 2, -2]]),
        Matrix([[1, 0, -1, 1]]),
        Matrix([[-1, 0, 2, -2]]),
        Matrix([[-1, 1, -1, 1]]),
        Matrix([[1, 0, 0, -1]]),
        Matrix([[-1, 1, 0, -1]]),
        Matrix([[0, -1, 1, 1]]),
        Matrix([[1, 1, -2, 0]]),
        Matrix([[-1, 2, -2, 0]]),
        Matrix([[0, -1, 2, -1]]),
        Matrix([[0, 0, -1, 2]]),
        Matrix([[2, -1, 0, 0]]),
        Matrix([[0, 0, 0, 0]]),
        Matrix([[0, 0, 0, 0]]),
        Matrix([[0, 0, 0, 0]]),
        Matrix([[0, 0, 0, 0]]),
        Matrix([[-2, 1, 0, 0]]),
        Matrix([[0, 0, 1, -2]]),
        Matrix([[0, 1, -2, 1]]),
        Matrix([[1, -2, 2, 0]]),
        Matrix([[-1, -1, 2, 0]]),
        Matrix([[0, 1, -1, -1]]),
        Matrix([[1, -1, 0, 1]]),
        Matrix([[-1, 0, 0, 1]]),
        Matrix([[1, -1, 1, -1]]),
        Matrix([[1, 0, -2, 2]]),
        Matrix([[-1, 0, 1, -1]]),
        Matrix([[-1, 1, -2, 2]]),
        Matrix([[1, 0, -1, 0]]),
        Matrix([[-1, 1, -1, 0]]),
        Matrix([[0, -1, 0, 2]]),
        Matrix([[1, 0, 0, -2]]),
        Matrix([[-1, 1, 0, -2]]),
        Matrix([[0, -1, 1, 0]]),
        Matrix([[0, -1, 2, -2]]),
        Matrix([[0, 0, -1, 1]]),
        Matrix([[0, 0, 0, -1]]),
        Matrix([[0, 1, -2, 0]]),
        Matrix([[1, -1, 0, 0]]),
        Matrix([[-1, 0, 0, 0]]),
    ]


def test_E6():
    E6 = E(6)

    # test subclass items
    assert E6.dimension == 6
    assert E6.n_pos_roots == 36

    assert E6.simple_roots == [
        Matrix([[S.Half, -S.Half, -S.Half, -S.Half, -
                 S.Half, -S.Half, -S.Half, S.Half]]),
        Matrix([[-1, 1, 0, 0, 0, 0, 0, 0]]),
        Matrix([[0, -1, 1, 0, 0, 0, 0, 0]]),
        Matrix([[0, 0, -1, 1, 0, 0, 0, 0]]),
        Matrix([[0, 0, 0, -1, 1, 0, 0, 0]]),
        Matrix([[1, 1, 0, 0, 0, 0, 0, 0]]),
    ]

    fw = E6.fundamental_weights[0]
    assert fw.basis == Basis.ORTHO
    assert E6.to_omega(fw) == Matrix([[1, 0, 0, 0, 0, 0]])


    # baseclass generated
    assert E6.cartan_matrix == Matrix([
        [2, -1, 0, 0, 0, 0],
        [-1, 2, -1, 0, 0, 0],
        [0, -1, 2, -1, 0, -1],
        [0, 0, -1, 2, -1, 0],
        [0, 0, 0, -1, 2, 0],
        [0, 0, -1, 0, 0, 2]])

    assert E6.omega_matrix == Matrix(
        [[0, 0, 0, 0, 0, Rational(-2, 3), Rational(-2, 3), Rational(2, 3)],
         [Rational(-1, 2), Rational(1, 2), Rational(1, 2), Rational(1, 2),
          Rational(1, 2), Rational(-5, 6), Rational(-5, 6), Rational(5, 6)],
            [0, 0, 1, 1, 1, -1, -1, 1],
            [0, 0, 0, 1, 1, Rational(-2, 3), Rational(-2, 3), Rational(2, 3)],
            [0, 0, 0, 0, 1, Rational(-1, 3), Rational(-1, 3), Rational(1, 3)],
         [Rational(1, 2), Rational(1, 2), Rational(1, 2), Rational(1, 2), Rational(1, 2), Rational(-1, 2), Rational(-1, 2), Rational(1, 2)], ])
    assert E6.metric_tensor == Matrix(
        [[Rational(4, 3), Rational(5, 3), 2, Rational(4, 3), Rational(2, 3), 1],
         [Rational(5, 3), Rational(10, 3), 4,
          Rational(8, 3), Rational(4, 3), 2],
            [2, 4, 6, 4, 2, 3],
            [Rational(4, 3), Rational(8, 3), 4,
             Rational(10, 3), Rational(5, 3), 2],
            [Rational(2, 3), Rational(4, 3), 2,
             Rational(5, 3), Rational(4, 3), 1],
         [1, 2, 3, 2, 1, 2], ])

    # backend
    assert E6.root_system() == [
        Matrix([[0, 0, 0, 0, 0, 1]]),
        Matrix([[0, 0, 1, 0, 0, -1]]),
        Matrix([[0, 1, -1, 1, 0, 0]]),
        Matrix([[0, 1, 0, -1, 1, 0]]),
        Matrix([[1, -1, 0, 1, 0, 0]]),
        Matrix([[-1, 0, 0, 1, 0, 0]]),
        Matrix([[0, 1, 0, 0, -1, 0]]),
        Matrix([[1, -1, 1, -1, 1, 0]]),
        Matrix([[-1, 0, 1, -1, 1, 0]]),
        Matrix([[1, -1, 1, 0, -1, 0]]),
        Matrix([[1, 0, -1, 0, 1, 1]]),
        Matrix([[-1, 0, 1, 0, -1, 0]]),
        Matrix([[-1, 1, -1, 0, 1, 1]]),
        Matrix([[1, 0, -1, 1, -1, 1]]),
        Matrix([[1, 0, 0, 0, 1, -1]]),
        Matrix([[-1, 1, -1, 1, -1, 1]]),
        Matrix([[-1, 1, 0, 0, 1, -1]]),
        Matrix([[0, -1, 0, 0, 1, 1]]),
        Matrix([[1, 0, 0, -1, 0, 1]]),
        Matrix([[1, 0, 0, 1, -1, -1]]),
        Matrix([[-1, 1, 0, -1, 0, 1]]),
        Matrix([[-1, 1, 0, 1, -1, -1]]),
        Matrix([[0, -1, 0, 1, -1, 1]]),
        Matrix([[0, -1, 1, 0, 1, -1]]),
        Matrix([[1, 0, 1, -1, 0, -1]]),
        Matrix([[-1, 1, 1, -1, 0, -1]]),
        Matrix([[0, -1, 1, -1, 0, 1]]),
        Matrix([[0, -1, 1, 1, -1, -1]]),
        Matrix([[0, 0, -1, 1, 1, 0]]),
        Matrix([[1, 1, -1, 0, 0, 0]]),
        Matrix([[-1, 2, -1, 0, 0, 0]]),
        Matrix([[0, -1, 2, -1, 0, -1]]),
        Matrix([[0, 0, -1, 0, 0, 2]]),
        Matrix([[0, 0, -1, 2, -1, 0]]),
        Matrix([[0, 0, 0, -1, 2, 0]]),
        Matrix([[2, -1, 0, 0, 0, 0]]),
        Matrix([[0, 0, 0, 0, 0, 0]]),
        Matrix([[0, 0, 0, 0, 0, 0]]),
        Matrix([[0, 0, 0, 0, 0, 0]]),
        Matrix([[0, 0, 0, 0, 0, 0]]),
        Matrix([[0, 0, 0, 0, 0, 0]]),
        Matrix([[0, 0, 0, 0, 0, 0]]),
        Matrix([[-2, 1, 0, 0, 0, 0]]),
        Matrix([[0, 0, 0, 1, -2, 0]]),
        Matrix([[0, 0, 1, -2, 1, 0]]),
        Matrix([[0, 0, 1, 0, 0, -2]]),
        Matrix([[0, 1, -2, 1, 0, 1]]),
        Matrix([[1, -2, 1, 0, 0, 0]]),
        Matrix([[-1, -1, 1, 0, 0, 0]]),
        Matrix([[0, 0, 1, -1, -1, 0]]),
        Matrix([[0, 1, -1, -1, 1, 1]]),
        Matrix([[0, 1, -1, 1, 0, -1]]),
        Matrix([[1, -1, -1, 1, 0, 1]]),
        Matrix([[-1, 0, -1, 1, 0, 1]]),
        Matrix([[0, 1, -1, 0, -1, 1]]),
        Matrix([[0, 1, 0, -1, 1, -1]]),
        Matrix([[1, -1, 0, -1, 1, 1]]),
        Matrix([[1, -1, 0, 1, 0, -1]]),
        Matrix([[-1, 0, 0, -1, 1, 1]]),
        Matrix([[-1, 0, 0, 1, 0, -1]]),
        Matrix([[0, 1, 0, 0, -1, -1]]),
        Matrix([[1, -1, 0, 0, -1, 1]]),
        Matrix([[1, -1, 1, -1, 1, -1]]),
        Matrix([[-1, 0, 0, 0, -1, 1]]),
        Matrix([[-1, 0, 1, -1, 1, -1]]),
        Matrix([[1, -1, 1, 0, -1, -1]]),
        Matrix([[1, 0, -1, 0, 1, 0]]),
        Matrix([[-1, 0, 1, 0, -1, -1]]),
        Matrix([[-1, 1, -1, 0, 1, 0]]),
        Matrix([[1, 0, -1, 1, -1, 0]]),
        Matrix([[-1, 1, -1, 1, -1, 0]]),
        Matrix([[0, -1, 0, 0, 1, 0]]),
        Matrix([[1, 0, 0, -1, 0, 0]]),
        Matrix([[-1, 1, 0, -1, 0, 0]]),
        Matrix([[0, -1, 0, 1, -1, 0]]),
        Matrix([[0, -1, 1, -1, 0, 0]]),
        Matrix([[0, 0, -1, 0, 0, 1]]),
        Matrix([[0, 0, 0, 0, 0, -1]]),
    ]

    assert E6.positive_roots == [
        Matrix([[0, 0, 0, 0, 0, 1]]),
        Matrix([[0, 0, 1, 0, 0, -1]]),
        Matrix([[0, 1, -1, 1, 0, 0]]),
        Matrix([[0, 1, 0, -1, 1, 0]]),
        Matrix([[1, -1, 0, 1, 0, 0]]),
        Matrix([[-1, 0, 0, 1, 0, 0]]),
        Matrix([[0, 1, 0, 0, -1, 0]]),
        Matrix([[1, -1, 1, -1, 1, 0]]),
        Matrix([[-1, 0, 1, -1, 1, 0]]),
        Matrix([[1, -1, 1, 0, -1, 0]]),
        Matrix([[1, 0, -1, 0, 1, 1]]),
        Matrix([[-1, 0, 1, 0, -1, 0]]),
        Matrix([[-1, 1, -1, 0, 1, 1]]),
        Matrix([[1, 0, -1, 1, -1, 1]]),
        Matrix([[1, 0, 0, 0, 1, -1]]),
        Matrix([[-1, 1, -1, 1, -1, 1]]),
        Matrix([[-1, 1, 0, 0, 1, -1]]),
        Matrix([[0, -1, 0, 0, 1, 1]]),
        Matrix([[1, 0, 0, -1, 0, 1]]),
        Matrix([[1, 0, 0, 1, -1, -1]]),
        Matrix([[-1, 1, 0, -1, 0, 1]]),
        Matrix([[-1, 1, 0, 1, -1, -1]]),
        Matrix([[0, -1, 0, 1, -1, 1]]),
        Matrix([[0, -1, 1, 0, 1, -1]]),
        Matrix([[1, 0, 1, -1, 0, -1]]),
        Matrix([[-1, 1, 1, -1, 0, -1]]),
        Matrix([[0, -1, 1, -1, 0, 1]]),
        Matrix([[0, -1, 1, 1, -1, -1]]),
        Matrix([[0, 0, -1, 1, 1, 0]]),
        Matrix([[1, 1, -1, 0, 0, 0]]),
        Matrix([[-1, 2, -1, 0, 0, 0]]),
        Matrix([[0, -1, 2, -1, 0, -1]]),
        Matrix([[0, 0, -1, 0, 0, 2]]),
        Matrix([[0, 0, -1, 2, -1, 0]]),
        Matrix([[0, 0, 0, -1, 2, 0]]),
        Matrix([[2, -1, 0, 0, 0, 0]]),
    ]
