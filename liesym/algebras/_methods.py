from __future__ import annotations

from enum import Enum

from sympy import acos, eye, Matrix, pi, sqrt, zeros


class Basis(Enum):
    """In literature there are several different basis representations
    used for convenience of calculations.

    ORTHO: The orthogonal basis which shows non-orthogonality between the simple roots.
    OMEGA: The omega basis, also known as the dynkin basis, is the basis of the fundamental weights.
    ALPHA: The alpha basis is the basis of the simple roots.
    """

    ORTHO = 0
    OMEGA = 1
    ALPHA = 2
    UNDEF = None


def _basis_lookup(x):
    if isinstance(x, Basis):
        return x
    if isinstance(x, str):
        if x.lower() == "ortho":
            return Basis.ORTHO
        if x.lower() == "omega":
            return Basis.OMEGA
        if x.lower() == "alpha":
            return Basis.ALPHA
        raise ValueError(
            "Unnsupported basis, string choices are 'ortho','alpha','omega'"
        )
    if x is None:
        return Basis.UNDEF

    raise ValueError("Unnsupported basis, string choices are 'ortho','alpha','omega'")


def annotate_matrix(M, basis=Basis.ORTHO):
    if getattr(M, "basis", None) is None:
        proper_basis = _basis_lookup(basis)
        setattr(M, "basis", proper_basis)

    if not isinstance(M.basis, Basis):
        proper_basis = _basis_lookup(M.basis)
        setattr(M, "basis", proper_basis)

    return M


def cartan_matrix(simple_roots: list[Matrix]) -> Matrix:
    rank = len(simple_roots)
    cartan_matrix = zeros(rank, rank)
    for i, sr_i in enumerate(simple_roots):
        for j, sr_j in enumerate(simple_roots):
            cartan_matrix[j, i] = 2 * sr_i.dot(sr_j) / sr_i.dot(sr_i)
    return cartan_matrix


def cocartan_matrix(simple_roots: list[Matrix]) -> Matrix:
    return Matrix([2 * x / x.dot(x) for x in simple_roots])


def quadratic_form(cartan_matrix: Matrix, simple_roots: list[Matrix]) -> Matrix:
    rank = len(simple_roots)
    quadratic_form = zeros(rank, rank)

    # normalized constant
    n_constant = sqrt(2 / max(x.dot(x) for x in simple_roots))
    for i in range(rank):
        root = n_constant * simple_roots[i]
        quadratic_form[i, i] = root.dot(root) / 2

    return cartan_matrix.pinv() * quadratic_form


def reflection_matrix(v):
    return (eye(len(v)) - 2 * v.T * v / v.dot(v)).as_immutable()


def reflection_matricies(simple_roots: list[Matrix]) -> list[Matrix]:
    return [reflection_matrix(x) for x in simple_roots]


def root_angle(a: Matrix, b: Matrix):
    """Calculates the angle between two roots a and b.

    Args:
        a (Matrix): A vector like matrix
        b (Matrix): A vector like matrix

    Returns:
        Integer: Resulting angle

    Examples
    ========
    >>> import liesym as ls
    >>> sr = ls.C(3).simple_roots
    >>> assert ls.root_angle(sr[0], sr[1]) == 120
    """
    return acos(a.dot(b) / sqrt(a.dot(a) * b.dot(b))) * 360 / (2 * pi)
