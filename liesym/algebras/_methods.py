from typing import List
from numpy.lib.arraysetops import isin
from sympy import zeros, Matrix, eye, sqrt
from enum import Enum


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
            "Unnsupported basis, string choices are 'ortho','alpha','omega'")
    if x is None:
        return Basis.UNDEF

    raise ValueError(
        "Unnsupported basis, string choices are 'ortho','alpha','omega'")


def _annotate_matrix(M, basis=Basis.ORTHO):
    if getattr(M, "basis", None) is None:
        proper_basis = _basis_lookup(basis)
        setattr(M, "basis", proper_basis)

    if not isinstance(M.basis, Basis):
        proper_basis = _basis_lookup(M.basis)
        setattr(M, "basis", proper_basis)

    return M


def _cartan_matrix(simple_roots: List[Matrix]) -> Matrix:
    rank = len(simple_roots)
    cartan_matrix = zeros(rank, rank)
    for i, sr_i in enumerate(simple_roots):
        for j, sr_j in enumerate(simple_roots):
            cartan_matrix[j, i] = 2 * sr_i.dot(sr_j) / sr_i.dot(sr_i)
    return cartan_matrix


def _cocartan_matrix(simple_roots: List[Matrix]) -> Matrix:
    return Matrix([2 * x / x.dot(x) for x in simple_roots])


def _quadratic_form(cartan_matrix: Matrix, simple_roots: List[Matrix]) -> Matrix:
    rank = len(simple_roots)
    quadratic_form = zeros(rank, rank)

    # normalized constant
    n_constant = sqrt(2 / max(x.dot(x) for x in simple_roots))
    for i in range(rank):
        root = n_constant * simple_roots[i]
        quadratic_form[i, i] = root.dot(root) / 2

    return cartan_matrix.pinv() * quadratic_form


def _reflection_matricies(simple_roots: List[Matrix]) -> List[Matrix]:
    def reflection_matrix(v): return (
        eye(len(v)) - 2 * v.T * v / v.dot(v)).as_immutable()
    return [reflection_matrix(x) for x in simple_roots]
