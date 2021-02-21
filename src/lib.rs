mod debug;
mod utils;
mod matrix_methods;
mod liealgebra_interface;

use crate::debug::{enable_debug};
use crate::liealgebra_interface::LieAlgebraBackend;


use pyo3::prelude::{pymodule, PyModule, PyResult, Python};

#[pymodule]
fn liesym(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    #[pyfn(m, "debug_mode")]
    fn debug_mode<'py>(_py: Python<'py>, is_on: bool) {
        if is_on {
            enable_debug();
        }
    }

    #[pyfn(m, "test")]
    fn test_none<'py>(_py: Python<'py>, num: Option<i64>) {
        match num {
            None => println!("None passed"),
            Some(x) => println!("{} passed", x)
        }
    }
    m.add_class::<LieAlgebraBackend>()?;



    // #[pyfn(m, "tensordecomposition")]
    // fn tensor_decomposition<'py>(
    //     py: Python<'py>,
    //     simple_roots: PyReadonlyArray3<i64>,
    //     cartan: PyReadonlyArray3<i64>,
    //     cartan_inv: PyReadonlyArray3<i64>,
    //     cocartan_t: PyReadonlyArray3<i64>,
    //     omega: PyReadonlyArray3<i64>,
    //     omega_inv: PyReadonlyArray3<i64>,
    //     n_roots: usize,
    //     rank: usize,
    //     irreps: PyReadonlyArray3<i64>,
    // ) ->  (&'py PyArray3<i64>, &'py PyArray3<i64>){
    //     if debug_on() {
    //         println!("Entering rust");
    //     }
    //     let methods = TensorDecomposition::new(
    //         simple_roots,
    //         cartan,
    //         cartan_inv,
    //         cocartan_t,
    //         omega,
    //         omega_inv,
    //         n_roots,
    //         rank);
    //     if debug_on() {
    //         println!("Tensor decomp constructed");
    //     }
    //     // drop(methods);
    //     let results = methods.tensor_product_decomp(to_rational_list(irreps));

    //     let (numer, denom) = vecarray_to_pyreturn(results);
    //     if debug_on() {
    //         println!("Returning to python");
    //     }
    //     (numer.into_pyarray(py), denom.into_pyarray(py))

    // }



    Ok(())
}
