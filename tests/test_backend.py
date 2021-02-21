import numpy as np

from liesym.algebras.backend import _LieAlgebraBackendWrapped
from liesym.algebras import A, B, C, D




def testBackendA():
    algebra = A(3)

    args = (
        algebra.rank,
        algebra.roots,
        algebra.simple_roots,
        algebra.cartan_matrix,
        algebra.cartan_matrix.pinv(),
        algebra.omega_matrix,
        algebra.omega_matrix.pinv(),
        algebra.cocartan_matrix,
    )

    def _test_backend(
        rank,
        roots,
        simple_roots,
        cartan_matrix,
        cartan_matrix_inverse,
        omega_matrix,
        omega_matrix_inverse,
        cocartan_matrix
    ):
        assert isinstance(rank, int)
        assert isinstance(roots, int)

        assert isinstance(simple_roots, np.ndarray)
        assert simple_roots.shape == (3,4,2)

        assert isinstance(cartan_matrix, np.ndarray)
        assert cartan_matrix.shape == (3,3,2)

        assert isinstance(cartan_matrix_inverse, np.ndarray)
        assert cartan_matrix_inverse.shape == (3,3,2)

        assert isinstance(omega_matrix, np.ndarray)
        assert omega_matrix.shape == (3,4,2)

        assert isinstance(omega_matrix_inverse, np.ndarray)
        assert omega_matrix_inverse.shape == (4,3,2)

        assert isinstance(cocartan_matrix, np.ndarray)
        assert cocartan_matrix.shape == (3,4,2)

    _LieAlgebraBackendWrapped(*args, backend=_test_backend)



def testBackendBCD():
    for cls in [B, C, D]:
        algebra = cls(3)

        args = (
            algebra.rank,
            algebra.roots,
            algebra.simple_roots,
            algebra.cartan_matrix,
            algebra.cartan_matrix.pinv(),
            algebra.omega_matrix,
            algebra.omega_matrix.pinv(),
            algebra.cocartan_matrix,
        )

        def _test_backend(
            rank,
            roots,
            simple_roots,
            cartan_matrix,
            cartan_matrix_inverse,
            omega_matrix,
            omega_matrix_inverse,
            cocartan_matrix
        ):
            assert isinstance(rank, int)
            assert rank == 3

            assert isinstance(roots, int)

            assert isinstance(simple_roots, np.ndarray)
            assert simple_roots.shape == (3,3,2)

            assert isinstance(cartan_matrix, np.ndarray)
            assert cartan_matrix.shape == (3,3,2)

            assert isinstance(cartan_matrix_inverse, np.ndarray)
            assert cartan_matrix_inverse.shape == (3,3,2)

            assert isinstance(omega_matrix, np.ndarray)
            assert omega_matrix.shape == (3,3,2)

            assert isinstance(omega_matrix_inverse, np.ndarray)
            assert omega_matrix_inverse.shape == (3,3,2)

            assert isinstance(cocartan_matrix, np.ndarray)
            assert cocartan_matrix.shape == (3,3,2)

        _LieAlgebraBackendWrapped(*args, backend=_test_backend)