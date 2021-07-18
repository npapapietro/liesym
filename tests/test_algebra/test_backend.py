import numpy as np
from sympy import S, Rational, Matrix

from liesym.algebras._backend import _LieAlgebraBackendWrapped
from liesym.algebras import A, B, C, D




def testBackendA():
    algebra = A(3)

    args = (
        algebra.rank,
        algebra.n_pos_roots,
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


        def orb(w,s):
            assert isinstance(w, np.ndarray)
            assert isinstance(s, np.ndarray)
            return (np.array([[[1,1],[2,2]]]), np.array([[[2,2],[3,3]]]))
        
        _test_backend.root_system = lambda: (np.array([[[1,1],[2,2]]]), np.array([[[2,2],[3,3]]]))
        _test_backend.orbit = orb
        _test_backend.tensor_product_decomposition = lambda x,_: orb(x,  np.array([]))
        _test_backend.irrep_by_dim = lambda *_:  (np.array([[[1,1],[2,2]]]), np.array([[[2,2],[3,3]]]))

        return _test_backend


    obj = _LieAlgebraBackendWrapped(*args, backend=_test_backend)
    expected = [Matrix([[S.Half, S.Half]]), Matrix([[Rational(2,3), Rational(2,3)]])]

    assert obj.root_system() == expected
    assert obj.orbit(expected[0], [1,2,3]) == expected
    assert obj.tensor_product_decomposition(*expected) == expected
    assert obj.get_irrep_by_dim(1,2) == expected



def testBackendBCD():
    for cls in [B, C, D]:
        algebra = cls(3)

        args = (
            algebra.rank,
            algebra.n_pos_roots,
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