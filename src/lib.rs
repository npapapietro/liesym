mod debug;
mod liealgebras;
mod matrix_methods;
mod utils;

use crate::debug::enable_debug;
use crate::liealgebras::LieAlgebraBackend;

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
            Some(x) => println!("{} passed", x),
        }
    }
    m.add_class::<LieAlgebraBackend>()?;

    Ok(())
}
