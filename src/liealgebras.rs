use numpy::{IntoPyArray, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray3};
use pyo3::prelude::{pyclass, pymethods, Python};
use rootsystem::RootSystem;

use crate::utils::{
    arrayr_to_pyreturn, to_rational_list, to_rational_matrix, to_rational_vector,
    vecarray_to_pyreturn,
};

type PyMat<'a> = PyReadonlyArray3<'a, i64>;
type PyReturn<'py> = (&'py PyArray3<i64>, &'py PyArray3<i64>);

#[pyclass]
pub struct LieAlgebraBackend {
    roots: RootSystem,
}

#[pymethods]
impl LieAlgebraBackend {
    #[new]
    fn new(
        rank: usize,
        roots: usize,
        simple_roots: PyMat,
        cartan_matrix_inverse: PyMat,
        omega_matrix: PyMat,
        omega_matrix_inverse: PyMat,
    ) -> Self {
        LieAlgebraBackend {
            roots: RootSystem::new(
                rank,
                roots,
                to_rational_list(simple_roots),
                to_rational_matrix(cartan_matrix_inverse),
                to_rational_matrix(omega_matrix),
                to_rational_matrix(omega_matrix_inverse),
            ),
        }
    }

    fn orbit<'py>(
        &self,
        py: Python<'py>,
        weight: PyMat,
        stabilizers: Option<PyReadonlyArray1<usize>>,
    ) -> PyReturn<'py> {
        let w = to_rational_vector(weight);
        let stabs = match stabilizers {
            Some(x) => Some(x.as_array().to_vec()),
            None => None,
        };
        let result = self.roots.orbit(w, stabs);

        let (numer, denom) = vecarray_to_pyreturn(result);

        (numer.into_pyarray(py), denom.into_pyarray(py))
    }

    /// Returns the root system of the algebra. The total number
    /// should be 2*N_pos_roots + rank
    fn root_system<'py>(&self, py: Python<'py>) -> PyReturn<'py> {
        let results = self.roots.root_system();
        let (numer, denom) = vecarray_to_pyreturn(results);
        (numer.into_pyarray(py), denom.into_pyarray(py))
    }

    /// Calculates the dimension of a irreducible representation
    fn dim<'py>(&self, _py: Python<'py>, irrep: PyReadonlyArray3<i64>) -> i64 {
        self.roots.dim(to_rational_vector(irrep))
    }

    fn irrep_by_dim<'py>(
        &self,
        py: Python<'py>,
        dim: i64,
        max_dynkin_digit: i64,
    ) -> Option<(&'py PyArray3<i64>, &'py PyArray3<i64>)> {
        let results = self.roots.irrep_by_dim(dim, max_dynkin_digit);
        if results.len() == 0 {
            return None;
        } else {
            let (n, d) = vecarray_to_pyreturn(results);
            return Some((n.into_pyarray(py), d.into_pyarray(py)));
        };
    }
    fn index_irrep<'py>(
        &self,
        _py: Python<'py>,
        irrep: PyReadonlyArray3<i64>,
        dim: i64,
    ) -> (i64, i64) {
        let result = self.roots.index_irrep(&to_rational_vector(irrep), dim);
        (*result.numer(), *result.denom())
    }

    fn conjugate_irrep<'py>(
        &self,
        py: Python<'py>,
        irrep: PyReadonlyArray3<i64>,
    ) -> (&'py PyArray2<i64>, &'py PyArray2<i64>) {
        let result = to_rational_vector(irrep);
        let (numer, denom) = arrayr_to_pyreturn(self.roots.conjugate(result));
        (numer.into_pyarray(py), denom.into_pyarray(py))
    }

    /// Performs the tensor product decomposition between two
    /// irreducible representations.
    fn tensor_product_decomposition<'py>(
        &self,
        py: Python<'py>,
        irrep1: PyReadonlyArray3<i64>,
        irrep2: PyReadonlyArray3<i64>,
    ) -> (&'py PyArray3<i64>, &'py PyArray3<i64>) {
        let results = self
            .roots
            .tensor_product_decomp(to_rational_vector(irrep1), to_rational_vector(irrep2));
        let (numer, denom) = vecarray_to_pyreturn(results);
        (numer.into_pyarray(py), denom.into_pyarray(py))
    }
}
