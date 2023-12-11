use itertools::Itertools;
use ndarray::parallel::prelude::{IntoParallelIterator, ParallelIterator};
use num::rational::Ratio;
use numpy::{IntoPyArray, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray3};
use pyo3::prelude::{pyclass, pymethods, Python};
use rootsystem::{Array2R, RootSystem};

use crate::utils::{
    arrayr_to_pyreturn, to_rational_list, to_rational_matrix, to_rational_vector,
    vecarray_to_pyreturn, Rational, Tap,
};

type PyMat<'a> = PyReadonlyArray3<'a, i64>;
type PyReturn<'py> = (&'py PyArray3<i64>, &'py PyArray3<i64>);

#[pyclass]
pub struct LieAlgebraBackend {
    roots: RootSystem,
    root_system: Vec<Array2R>,
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
        let roots = RootSystem::new(
            rank,
            roots,
            to_rational_list(simple_roots),
            to_rational_matrix(cartan_matrix_inverse),
            to_rational_matrix(omega_matrix),
            to_rational_matrix(omega_matrix_inverse),
        );
        LieAlgebraBackend {
            root_system: roots.root_system(),
            roots,
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
        let irrep = to_rational_vector(irrep);
        self._dim(&irrep)
    }

    fn irrep_by_dim<'py>(
        &self,
        py: Python<'py>,
        dim: i64,
        max_dynkin_digit: i64,
    ) -> Option<(&'py PyArray3<i64>, &'py PyArray3<i64>)> {
        let results = self._irrep_by_dim(dim, max_dynkin_digit);

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
        let (numer, denom) = arrayr_to_pyreturn(self._conjugate(result));
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

impl LieAlgebraBackend {
    /// Calculates the dimension of a irreducible representation
    fn _dim(&self, irrep: &Array2R) -> i64 {
        let rho = Array2R::ones((1, self.roots.rank));

        let mut dim = Ratio::from(1);

        for root in self.get_postive_roots().iter() {
            dim *= self.roots.scalar_product(&(irrep + &rho), root)
                / self.roots.scalar_product(&rho, root);
        }

        dim.to_integer()
    }

    pub fn get_postive_roots(&self) -> Vec<Array2R> {
        self.root_system[..self.roots.n_roots].to_vec()
    }

    pub fn _irrep_by_dim(&self, dim: i64, max_dyn_digit: i64) -> Vec<Array2R> {
        (0..self.roots.rank)
            .map(|_| 0..(max_dyn_digit + 1))
            .multi_cartesian_product()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|i| i.to_ratio())
            .filter(|i| self._dim(&i) == dim)
            .collect::<Vec<_>>()
            .tap(|x| {
                x.sort_by(|a, b| {
                    let i1 = self.roots.index_irrep(a, dim);
                    let i2 = self.roots.index_irrep(b, dim);
                    i1.cmp(&i2)
                        .then(Vec::from_iter(b.iter()).cmp(&Vec::from_iter(a.iter())))
                })
            })
    }

    pub fn _conjugate(&self, irrep: Array2R) -> Array2R {
        let max = irrep.iter().max().unwrap().clone().to_integer() as usize;
        let dim = self._dim(&irrep);
        let idx = self.roots.index_irrep(&irrep, dim);
        match (0..self.roots.rank)
            .map(|_| 0..max + 1)
            .multi_cartesian_product()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|i| i.to_ratio())
            .filter(|x| {
                let d = self._dim(x);
                x.clone() != irrep && dim == d && idx == self.roots.index_irrep(x, d)
            })
            .collect::<Vec<_>>()
            .iter()
            .next()
        {
            Some(x) => x.clone(),
            None => irrep,
        }
    }
}
