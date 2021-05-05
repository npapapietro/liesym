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
        write!(f, "{}", self.formatter())
    }
}

impl Frac {
    fn formatter(&self) -> String {
        if self.0.numer() == &0 {
            "0".to_string()
        } else if self.0.denom() == &1 {
            format!("{}", self.0.numer())
        } else {
            format!("{}/{}", self.0.numer(), self.0.denom())
        }
    }

    #[allow(dead_code)]
    pub fn print<D: Dimension>(m: Array<Ratio<i64>, D>) {
        println!("{}", m.mapv(|x| Frac { 0: x }))
    }

    #[allow(dead_code)]
    pub fn print_vec_r<D: Dimension>(v: Vec<Array<Ratio<i64>, D>>) {
        for i in v.iter() {
            Frac::print(i.clone());
        }
    }
    #[allow(dead_code)]
    pub fn format<D: Dimension>(m: Array<Ratio<i64>, D>) -> String {
        "(".to_string()
            + &m.iter()
                .map(|x| Frac { 0: x.clone() }.formatter())
                .collect::<Vec<String>>()
                .join(",")
            + ")"
    }
}

#[allow(dead_code)]
pub struct Logger {}
impl Logger {
    #[allow(dead_code)]
    pub fn debug(msg: String) {
        if debug_on() {
            println!("{}", msg);
        }
    }
}
