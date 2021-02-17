use ndarray::{Array, Dimension};
use num::rational::Ratio;
use std::fmt::{Display, Formatter, Result};

static mut DEBUGMODE: bool = false;

pub fn enable_debug() {
    unsafe { DEBUGMODE = true }
}

pub fn debug_on() -> bool {
    unsafe { DEBUGMODE }
}

#[derive(Debug)]
pub struct Frac(Ratio<i64>);

impl Display for Frac {
    fn fmt(&self, f: &mut Formatter) -> Result {
        if self.0.numer() == &0 {
            write!(f, "0")
        } else if self.0.denom() == &1 {
            write!(f, "{}", self.0.numer())
        } else {
            write!(f, "{}/{}", self.0.numer(), self.0.denom())
        }
    }
}

impl Frac {
    #[allow(dead_code)]
    pub fn print<D>(m: Array<Ratio<i64>, D>)
    where
        D: Dimension,
    {
        println!("{}", m.mapv(|x| Frac { 0: x }))
    }
}
