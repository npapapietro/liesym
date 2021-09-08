from sympy import Matrix, I, LeviCivita, sympify

from liesym import SU, SO, Sp, A, B, C, D


def test_su():
    su2 = SU(2)

    assert su2.dimension == 2
    assert su2.group == "SU"

    assert su2.generators() == [
        Matrix([
            [0, 1],
            [1, 0]]) / 2,
        Matrix([
            [0, -I],
            [I,  0]]) / 2,
        Matrix([
            [1,  0],
            [0, -1]]) / 2]

    assert su2.algebra == A(1)

    for i in range(3):
        for j in range(3):
            for k in range(3):
                assert su2.structure_constants(i, j, k) == LeviCivita(i, j, k)

    for n in range(2, 5):
        g = SU(n)
        assert g.quadratic_casimir(n) == sympify(n**2 - 1) / sympify(2 * n)


def test_so():
    so3 = SO(3)

    assert so3.dimension == 3
    assert so3.group == "SO"

    assert so3.generators() == [
        Matrix([
            [0, I, 0],
            [-I, 0, 0],
            [0, 0, 0]]),
        Matrix([
            [0, 0, I],
            [0, 0, 0],
            [-I, 0, 0]]),
        Matrix([
            [0,  0, 0],
            [0,  0, I],
            [0, -I, 0]])]

    assert so3.generators(True) == [
        (Matrix([
            [0, I, 0],
            [-I, 0, 0],
            [0, 0, 0]]), (1, 0)),
        (Matrix([
            [0, 0, I],
            [0, 0, 0],
            [-I, 0, 0]]), (2, 0)),
        (Matrix([
            [0,  0, 0],
            [0,  0, I],
            [0, -I, 0]]), (2, 1))
    ]

    assert so3.algebra == B(1)

    assert SO(4).algebra == D(2)

    for n in range(5, 7):
        g = SO(n)
        r = g.algebra.fundamental_weights[0]
        assert g.quadratic_casimir(r) == sympify(n - 1) / 2


def test_sp():
    sp4 = Sp(4)

    assert sp4.dimension == 4
    assert sp4.group == "Sp"

    assert len(sp4.generators()) == 10

    assert sp4.algebra == C(2)

    for n in range(2, 5):
        g = Sp(2 * n)
        r = g.algebra.fundamental_weights[0]
        assert g.quadratic_casimir(r) == sympify(2 * n + 1) / 2
