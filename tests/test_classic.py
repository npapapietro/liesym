from liesym import A, B, C, D
from sympy import Matrix, Rational


def test_A():
    A2 = A(2)

    # test subclass items
    assert A2.dimension == 3
    assert A2.n_pos_roots == 3

    assert A2.simple_roots == [
        Matrix([[1, -1, 0]]),
        Matrix([[0, 1, -1]]),
    ]

    # baseclass generated
    A3 = A(3)
    assert A3.cartan_matrix == Matrix([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
    assert A3.cocartan_matrix == Matrix(
        [[1, -1, 0, 0], [0, 1, -1, 0], [0, 0, 1, -1]])
    assert A3.omega_matrix == Matrix([[Rational(3, 4), Rational(-1, 4), Rational(-1, 4), Rational(-1, 4)], [Rational(
        1, 2), Rational(1, 2), Rational(-1, 2), Rational(-1, 2)], [Rational(1, 4), Rational(1, 4), Rational(1, 4), Rational(-3, 4)]])
    assert A3.metric_tensor == Matrix([[Rational(3, 4), Rational(1, 2), Rational(1, 4)], [
                                      Rational(1, 2), 1, Rational(1, 2)], [Rational(1, 4), Rational(1, 2), Rational(3, 4)]])
    assert A3.reflection_matricies == [
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
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]])]
    assert A3.fundamental_weights == [
        Matrix([[Rational(3, 4), Rational(-1, 4), Rational(-1, 4), Rational(-1, 4)]]),
        Matrix([[Rational(1, 2), Rational(1, 2), Rational(-1, 2), Rational(-1, 2)]]),
        Matrix([[Rational(1, 4), Rational(1, 4), Rational(1, 4), Rational(-3, 4)]])]

    # backend
    assert A3.root_system() == [
        Matrix([[1, 0, 1]]),
        Matrix([[-1, 1, 1]]),
        Matrix([[1, 1, -1]]),
        Matrix([[-1, 2, -1]]),
        Matrix([[0, -1, 2]]),
        Matrix([[2, -1, 0]]),
        Matrix([[0, 0, 0]]),
        Matrix([[0, 0, 0]]),
        Matrix([[0, 0, 0]]),
        Matrix([[-2, 1, 0]]),
        Matrix([[0, 1, -2]]),
        Matrix([[1, -2, 1]]),
        Matrix([[-1, -1, 1]]),
        Matrix([[1, -1, -1]]),
        Matrix([[-1, 0, -1]])
    ]

    assert A3.positive_roots == [
        Matrix([[1, 0, 1]]),
        Matrix([[-1, 1, 1]]),
        Matrix([[1, 1, -1]]),
        Matrix([[-1, 2, -1]]),
        Matrix([[0, -1, 2]]),
        Matrix([[2, -1, 0]]),
    ]

    fund = Matrix([[1, 0, 0]])
    antifund = Matrix([[0, 0, 1]])
    decomp = A3.tensor_product_decomposition([fund, antifund])

    assert set([x.as_immutable() for x in decomp]) == set([
        Matrix([[1, 0, 1]]).as_immutable(),
        Matrix([[0, 0, 0]]).as_immutable(),
    ])

    assert A3.dim(fund) == 4

    adj = Matrix([[1, 0, 1]])
    assert A3.dim(adj) == 15

    assert A3.max_dynkin_digit(adj) == 2

    assert A3.get_irrep_by_dim(15) == [adj]

    assert A3.conjugate(Matrix([[1,1,0]])) == Matrix([[0,1,1]])

def test_A1():
    A2 = A(1)

    # test subclass items
    assert A2.dimension == 2
    assert A2.root_system() == [Matrix([[2]]), Matrix([[0]]), Matrix([[-2]])]


def test_B():
    B2 = B(2)

    # test subclass items
    assert B2.dimension == 2
    assert B2.n_pos_roots == 4

    assert B2.simple_roots == [
        Matrix([[1, -1]]),
        Matrix([[0, 1]]),
    ]

    # baseclass generated
    B3 = B(3)
    assert B3.cartan_matrix == Matrix([[2, -1, 0], [-1, 2, -2], [0, -1, 2]])
    assert B3.cocartan_matrix == Matrix([[1, -1, 0], [0, 1, -1], [0, 0, 2]])
    assert B3.omega_matrix == Matrix(
        [[1, 0, 0], [1, 1, 0], [Rational(1, 2), Rational(1, 2), Rational(1, 2)]])
    assert B3.metric_tensor == Matrix([[1, 1, Rational(1, 2)], [1, 2, 1], [
                                      Rational(1, 2), 1, Rational(3, 4)]])
    assert B3.reflection_matricies == [
        Matrix([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1]]),
        Matrix([
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 0]]),
        Matrix([
            [1, 0,  0],
            [0, 1,  0],
            [0, 0, -1]])]
    assert B3.fundamental_weights == [
        Matrix([[1, 0, 0]]),
        Matrix([[1, 1, 0]]),
        Matrix([[Rational(1, 2), Rational(1, 2), Rational(1, 2)]])
    ]

    # backend
    assert B3.root_system() == [
        Matrix([[0, 1, 0]]),
        Matrix([[1, -1, 2]]),
        Matrix([[-1, 0, 2]]),
        Matrix([[1, 0, 0]]),
        Matrix([[-1, 1, 0]]),
        Matrix([[1, 1, -2]]),
        Matrix([[-1, 2, -2]]),
        Matrix([[0, -1, 2]]),
        Matrix([[2, -1, 0]]),
        Matrix([[0, 0, 0]]),
        Matrix([[0, 0, 0]]),
        Matrix([[0, 0, 0]]),
        Matrix([[-2, 1, 0]]),
        Matrix([[0, 1, -2]]),
        Matrix([[1, -2, 2]]),
        Matrix([[-1, -1, 2]]),
        Matrix([[1, -1, 0]]),
        Matrix([[-1, 0, 0]]),
        Matrix([[1, 0, -2]]),
        Matrix([[-1, 1, -2]]),
        Matrix([[0, -1, 0]]),
    ]

    assert B3.positive_roots == [
        Matrix([[0, 1, 0]]),
        Matrix([[1, -1, 2]]),
        Matrix([[-1, 0, 2]]),
        Matrix([[1, 0, 0]]),
        Matrix([[-1, 1, 0]]),
        Matrix([[1, 1, -2]]),
        Matrix([[-1, 2, -2]]),
        Matrix([[0, -1, 2]]),
        Matrix([[2, -1, 0]]),
    ]

    decomp = B3.tensor_product_decomposition(
        [Matrix([[1, 0, 0]]), Matrix([[1, 0, 0]]), Matrix([[1, 0, 0]])])
    assert sorted([tuple(x.tolist()) for x in decomp]) == sorted([
        tuple(x.tolist()) for x in [
            Matrix([[1, 0, 0]]),
            Matrix([[1, 0, 0]]),
            Matrix([[1, 0, 0]]),
            Matrix([[3, 0, 0]]),
            Matrix([[0, 0, 2]]),
            Matrix([[1, 1, 0]]),
            Matrix([[1, 1, 0]]),
        ]])

    adj = Matrix([[1, 0, 1]])
    assert B3.dim(adj) == 48

    assert B3.max_dynkin_digit(adj) == 2

    assert B3.get_irrep_by_dim(48) == [adj]


def test_C():
    C2 = C(2)

    # test subclass items
    assert C2.dimension == 2
    assert C2.n_pos_roots == 4

    assert C2.simple_roots == [
        Matrix([[1, -1]]),
        Matrix([[0, 2]]),
    ]

    # baseclass generated
    C3 = C(3)
    assert C3.cartan_matrix == Matrix([[2, -1, 0], [-1, 2, -1], [0, -2, 2]])
    assert C3.cocartan_matrix == Matrix([[1, -1, 0], [0, 1, -1], [0, 0, 1]])
    assert C3.omega_matrix == Matrix([[1, 0, 0], [1, 1, 0], [1, 1, 1]])
    assert C3.metric_tensor == Matrix([[Rational(1, 2), Rational(1, 2), Rational(
        1, 2)], [Rational(1, 2), 1, 1], [Rational(1, 2), 1, Rational(3, 2)]])
    assert C3.reflection_matricies == [
        Matrix([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1]]),
        Matrix([
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 0]]),
        Matrix([
            [1, 0,  0],
            [0, 1,  0],
            [0, 0, -1]])]
    assert C3.fundamental_weights == [
        Matrix([[1, 0, 0]]),
        Matrix([[1, 1, 0]]),
        Matrix([[1, 1, 1]])
    ]
    # backend
    assert C3.root_system() == [
        Matrix([[2, 0, 0]]),
        Matrix([[0, 1, 0]]),
        Matrix([[-2, 2, 0]]),
        Matrix([[1, -1, 1]]),
        Matrix([[-1, 0, 1]]),
        Matrix([[1, 1, -1]]),
        Matrix([[-1, 2, -1]]),
        Matrix([[0, -2, 2]]),
        Matrix([[2, -1, 0]]),
        Matrix([[0, 0, 0]]),
        Matrix([[0, 0, 0]]),
        Matrix([[0, 0, 0]]),
        Matrix([[-2, 1, 0]]),
        Matrix([[0, 2, -2]]),
        Matrix([[1, -2, 1]]),
        Matrix([[-1, -1, 1]]),
        Matrix([[1, 0, -1]]),
        Matrix([[-1, 1, -1]]),
        Matrix([[2, -2, 0]]),
        Matrix([[0, -1, 0]]),
        Matrix([[-2, 0, 0]]),
    ]

    assert C3.positive_roots == [
        Matrix([[2, 0, 0]]),
        Matrix([[0, 1, 0]]),
        Matrix([[-2, 2, 0]]),
        Matrix([[1, -1, 1]]),
        Matrix([[-1, 0, 1]]),
        Matrix([[1, 1, -1]]),
        Matrix([[-1, 2, -1]]),
        Matrix([[0, -2, 2]]),
        Matrix([[2, -1, 0]]),
    ]

    adj = Matrix([[1, 0, 1]])
    assert C3.dim(adj) == 70
    assert C3.max_dynkin_digit(adj) == 2
    assert C3.get_irrep_by_dim(70) == [adj]


def test_D():
    D2 = D(2)

    # test subclass items
    assert D2.dimension == 2
    assert D2.n_pos_roots == 2

    assert D2.simple_roots == [
        Matrix([[1, -1]]),
        Matrix([[1, 1]]),
    ]

    # baseclass generated
    D3 = D(3)
    assert D3.cartan_matrix == Matrix([[2, -1, -1], [-1, 2, 0], [-1, 0, 2]])
    assert D3.cocartan_matrix == Matrix([[1, -1, 0], [0, 1, -1], [0, 1, 1]])
    assert D3.omega_matrix == Matrix([[1, 0, 0], [Rational(1, 2), Rational(
        1, 2), Rational(-1, 2)], [Rational(1, 2), Rational(1, 2), Rational(1, 2)]])
    assert D3.metric_tensor == Matrix([[1, Rational(1, 2), Rational(1, 2)], [Rational(
        1, 2), Rational(3, 4), Rational(1, 4)], [Rational(1, 2), Rational(1, 4), Rational(3, 4)]])
    assert D3.reflection_matricies == [
        Matrix([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1]]),
        Matrix([
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 0]]),
        Matrix([
            [1,  0,  0],
            [0,  0, -1],
            [0, -1,  0]])]
    assert D3.fundamental_weights == [
        Matrix([[1, 0, 0]]),
        Matrix([[Rational(1, 2), Rational(1, 2), Rational(-1, 2)]]),
        Matrix([[Rational(1, 2), Rational(1, 2), Rational(1, 2)]])
    ]

    # backend
    assert D3.root_system() == [
        Matrix([[0, 1, 1]]),
        Matrix([[1, -1, 1]]),
        Matrix([[1, 1, -1]]),
        Matrix([[-1, 0, 2]]),
        Matrix([[-1, 2, 0]]),
        Matrix([[2, -1, -1]]),
        Matrix([[0, 0, 0]]),
        Matrix([[0, 0, 0]]),
        Matrix([[0, 0, 0]]),
        Matrix([[-2, 1, 1]]),
        Matrix([[1, -2, 0]]),
        Matrix([[1, 0, -2]]),
        Matrix([[-1, -1, 1]]),
        Matrix([[-1, 1, -1]]),
        Matrix([[0, -1, -1]]),
    ]

    assert D3.positive_roots == [
        Matrix([[0, 1, 1]]),
        Matrix([[1, -1, 1]]),
        Matrix([[1, 1, -1]]),
        Matrix([[-1, 0, 2]]),
        Matrix([[-1, 2, 0]]),
        Matrix([[2, -1, -1]]),
    ]

    adj = Matrix([[1, 0, 1]])
    assert D3.dim(adj) == 20
    assert D3.max_dynkin_digit(adj) == 2
    assert D3.get_irrep_by_dim(20) == [Matrix([[1, 1, 0]]),
                                       Matrix([[1, 0, 1]]),
                                       Matrix([[2, 0, 0]]),
                                       Matrix([[0, 3, 0]]),
                                       Matrix([[0, 0, 3]])]

    D5 = D(5)
    assert D5.conjugate(Matrix([[1, 0, 0, 0, 0]])) ==  Matrix([[1, 0, 0, 0, 0]])
                                      