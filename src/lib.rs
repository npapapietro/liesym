mod debug;
mod orbit;
mod utils;
mod tensordecomp;
mod matrix_methods;

use crate::debug::{debug_on, enable_debug};
use crate::orbit::OrbitMethods;
use crate::utils::{to_rational_list, vecarray_to_pyreturn};
use crate::tensordecomp::TensorDecomposition;

use numpy::{IntoPyArray, PyArray3, PyReadonlyArray1, PyReadonlyArray3};
use pyo3::prelude::{pymodule, PyModule, PyResult, Python};

#[pymodule]
fn liesym(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    #[pyfn(m, "debug_mode")]
    fn test<'py>(_py: Python<'py>, is_on: bool) {
        if is_on {
            enable_debug();
        }
    }

    #[pyfn(m, "unstabilized_orbit")]
    fn orbit<'py>(
        py: Python<'py>,
        simple_roots: PyReadonlyArray3<i64>,
        omega_inv: PyReadonlyArray3<i64>,
        omega: PyReadonlyArray3<i64>,
        cartan_inv: PyReadonlyArray3<i64>,
        cocartan_t: PyReadonlyArray3<i64>,
        n_roots: usize,
        rank: usize,
        weight: PyReadonlyArray3<i64>,
    ) -> (&'py PyArray3<i64>, &'py PyArray3<i64>) {
        if debug_on() {
            println!("Entering rust");
        }
        let methods = OrbitMethods::new(simple_roots, omega_inv, omega,cartan_inv, cocartan_t, n_roots, rank);
        if debug_on() {
            println!("Orbits constructed");
        }
        let weight = to_rational_list(weight).first().unwrap().clone();
        let results = methods.orbit(weight);
        if debug_on() {
            println!("Orbits calculated");
        }
        let (numer, denom) = vecarray_to_pyreturn(results);
        if debug_on() {
            println!("Returning to python");
        }
        (numer.into_pyarray(py), denom.into_pyarray(py))
    }

    #[pyfn(m, "stabilized_orbit")]
    fn stabilized_orbit<'py>(
        py: Python<'py>,
        simple_roots: PyReadonlyArray3<i64>,
        omega_inv: PyReadonlyArray3<i64>,
        omega: PyReadonlyArray3<i64>,
        cartan_inv: PyReadonlyArray3<i64>,
        cocartan_t: PyReadonlyArray3<i64>,
        n_roots: usize,
        rank: usize,
        weight: PyReadonlyArray3<i64>,
        stabilizers: PyReadonlyArray1<usize>,
    ) -> (&'py PyArray3<i64>, &'py PyArray3<i64>) {
        if debug_on() {
            println!("Entering rust");
        }
        let methods = OrbitMethods::new(simple_roots, omega_inv, omega, cartan_inv, cocartan_t, n_roots, rank);
        if debug_on() {
            println!("Orbits constructed");
        }

        let results = methods.stable_orbit(
            to_rational_list(weight).first().unwrap().clone(),
            stabilizers.as_array().iter().map(|x| x.clone()).collect()
        );
        if debug_on() {
            println!("Orbits calculated");
        }
        let (numer, denom) = vecarray_to_pyreturn(results);
        if debug_on() {
            println!("Returning to python");
        }
        (numer.into_pyarray(py), denom.into_pyarray(py))
    }

    #[pyfn(m, "tensordecomposition")]
    fn tensor_decomposition<'py>(
        py: Python<'py>,
        simple_roots: PyReadonlyArray3<i64>,
        cartan: PyReadonlyArray3<i64>,
        cartan_inv: PyReadonlyArray3<i64>,
        cocartan_t: PyReadonlyArray3<i64>,
        omega: PyReadonlyArray3<i64>,
        omega_inv: PyReadonlyArray3<i64>,
        n_roots: usize,
        rank: usize,
        irreps: PyReadonlyArray3<i64>,
    ) ->  (&'py PyArray3<i64>, &'py PyArray3<i64>){
        if debug_on() {
            println!("Entering rust");
        }
        let methods = TensorDecomposition::new(
            simple_roots,
            cartan,
            cartan_inv,
            cocartan_t,
            omega,
            omega_inv,
            n_roots,
            rank);
        if debug_on() {
            println!("Tensor decomp constructed");
        }
        // drop(methods);
        let results = methods.tensor_product_decomp(to_rational_list(irreps));

        let (numer, denom) = vecarray_to_pyreturn(results);
        if debug_on() {
            println!("Returning to python");
        }
        (numer.into_pyarray(py), denom.into_pyarray(py))

    }



    Ok(())
}
