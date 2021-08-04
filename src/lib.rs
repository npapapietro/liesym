mod debug;
mod liealgebras;
mod matrix_methods;
mod utils;


use crate::debug::enable_debug;
use crate::liealgebras::{LieAlgebraBackend, struct_consts};
use crate::utils::to_complex_list;

use pyo3::prelude::{pymodule, PyModule, PyResult, Python};
use numpy::PyReadonlyArray4;

#[pymodule]
fn liesym(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    #[pyfn(m)]
    fn debug_mode<'py>(_py: Python<'py>, is_on: bool) {
        if is_on {
            enable_debug();
        }
    }

    #[pyfn(m)]
    fn structure_constants<'py>(py: Python<'py>, reals: PyReadonlyArray4<i64>, imags: PyReadonlyArray4<i64>){
        let generators = to_complex_list(reals, imags);

        // println!("{}, {:?}", reals_vec.len(), reals_vec[0].shape());
        
        struct_consts(&generators);
    }

    m.add_class::<LieAlgebraBackend>()?;

    Ok(())
}
