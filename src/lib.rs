mod debug;
mod liealgebras;
mod utils;

use crate::liealgebras::LieAlgebraBackend;
use pyo3::prelude::{pymodule, PyModule, PyResult, Python};

#[pymodule]
fn liesym(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<LieAlgebraBackend>()?;

    Ok(())
}
