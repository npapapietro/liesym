mod common;
mod roots;
pub use roots::RootSystem;

use ndarray::Array2;
use num::{rational::Ratio, Complex};

pub type Array2R = Array2<Ratio<i64>>;
pub type Array2C = Array2<Complex<Ratio<i64>>>;
