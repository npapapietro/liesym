mod debug;
mod liealgebras;
mod subalgebras;
mod utils;

use std::str::FromStr;

use crate::liealgebras::LieAlgebra;
use crate::subalgebras::{BranchingRule, LieGroup};
use log::LevelFilter;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::{pymodule, PyModule, PyResult, Python};
use pyo3::{pyfunction, wrap_pyfunction, PyErr};
use simple_logger::SimpleLogger;

#[pyfunction]
fn setup_logging(level: &str) -> PyResult<()> {
    let lvl =
        LevelFilter::from_str(level).map_err(|e| PyErr::new::<PyTypeError, _>(format!("{}", e)))?;
    SimpleLogger::new().with_level(lvl).init().unwrap();
    Ok(())
}

#[pymodule]
#[pyo3(name = "_liesym_rust")]
fn liesym(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<LieAlgebra>()?;
    m.add_class::<LieGroup>()?;
    m.add_class::<BranchingRule>()?;
    m.add_function(wrap_pyfunction!(setup_logging, m)?)?;

    Ok(())
}
