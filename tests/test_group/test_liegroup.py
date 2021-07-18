from sympy import Matrix, I

from liesym import SU, SO, Sp, A, B, C, D


def test_su():
    su2 = SU(2)

    assert su2.dimension == 2
    assert su2.group == "SU"

    assert su2.generators() == [
        Matrix([
            [0, 1],
            [1, 0]]),
        Matrix([
            [0, -I],
            [I,  0]]),
        Matrix([
            [1,  0],
            [0, -1]])]

    assert su2.algebra == A(1)


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


def test_sp():
    sp4 = Sp(4)

    assert sp4.dimension == 4
    assert sp4.group == "Sp"

    assert len(sp4.generators()) == 10

    assert sp4.algebra == C(2)
