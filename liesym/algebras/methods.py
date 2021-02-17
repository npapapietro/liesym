from typing import List
from sympy import zeros, Matrix, eye


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
    for i in range(rank):
        quadratic_form[i, i] = simple_roots[i].dot(simple_roots[i]) / 2

    return cartan_matrix.pinv() * quadratic_form


def _reflection_matricies(simple_roots: List[Matrix]) -> List[Matrix]:
    def reflection_matrix(v): return (
        eye(len(v)) - 2 * v.T * v / v.dot(v)).as_immutable()
    return [reflection_matrix(x) for x in simple_roots]


# def _tensor_decomp(
#         simple_roots: List[Matrix],
#         cartan: Matrix,
#         cartan_inv: Matrix,
#         omega: Matrix,
#         omega_inv: Matrix,
#         n_roots: int,
#         rank: int,
#         irreps: List[Matrix]):

#     args = (
#         _prepare_matrix(simple_roots),
#         _prepare_matrix(cartan),
#         _prepare_matrix(cartan_inv),
#         _prepare_matrix(omega),
#         _prepare_matrix(omega_inv),
#         n_roots,
#         rank,
#         _prepare_matrix(irreps),
#     )

#     return _reformat_orbits(_tensordecomposition(*args))
