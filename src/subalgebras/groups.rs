use std::fmt::{self, Display, Formatter};

use crate::utils::Tap;

use itertools::Itertools;
use pyo3::{exceptions::PyException, pyclass, pymethods, PyResult};

use super::BranchingRule;

pub mod group_types {
    pub type GroupType = &'static str;

    pub const SU: GroupType = "SU";
    pub const SO: GroupType = "SO";
    pub const SP: GroupType = "SP";
    pub const E6: GroupType = "E6";
    pub const E7: GroupType = "E7";
    pub const E8: GroupType = "E8";
    pub const G2: GroupType = "G2";
    pub const F4: GroupType = "F4";
    pub const U1: GroupType = "U1";
}
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[pyclass]
pub struct LieGroup {
    pub type_: group_types::GroupType,
    pub dim: usize,
}

// Constant dim groups
pub const U1: LieGroup = LieGroup {
    type_: group_types::U1,
    dim: 1,
};
pub const E6: LieGroup = LieGroup {
    type_: group_types::E6,
    dim: 6,
};
pub const E7: LieGroup = LieGroup {
    type_: group_types::E7,
    dim: 7,
};
pub const E8: LieGroup = LieGroup {
    type_: group_types::E8,
    dim: 8,
};
pub const G2: LieGroup = LieGroup {
    type_: group_types::G2,
    dim: 2,
};
pub const F4: LieGroup = LieGroup {
    type_: group_types::F4,
    dim: 4,
};

// Rust only implementations
impl LieGroup {
    pub fn new<T: ToString>(alg: T, d: Option<usize>) -> Result<Self, &'static str> {
        let dim = match d {
            Some(x) => Ok(x),
            None => Err("Dimension is required for this algebra"),
        };
        Ok(match alg.to_string().to_uppercase().as_str() {
            "SU" => match dim {
                Ok(1) => U1,
                Ok(n) => Self {
                    type_: group_types::SU,
                    dim: n,
                },
                Err(m) => Err(m)?,
            },
            "SO" => Self {
                type_: group_types::SO,
                dim: dim?,
            },
            "SP" => Self {
                type_: group_types::SP,
                dim: dim?,
            },
            "E6" => E6,
            "E7" => E7,
            "E8" => E8,
            "F4" => F4,
            "G2" => G2,
            "U1" => U1,
            _ => Err("Unknown algebra type or dimension unsupported")?,
        })
    }
}

impl Display for LieGroup {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self.type_ {
                group_types::SO | group_types::SU | group_types::SP => {
                    format!("{}({})", self.type_, self.dim)
                }
                _ => format!("{}", self.type_),
            }
        )
    }
}

// Python implementations
#[pymethods]
impl LieGroup {
    #[new]
    fn pynew(alg: String, dim: Option<usize>) -> PyResult<Self> {
        LieGroup::new(alg, dim).map_err(PyException::new_err)
    }

    fn maximal_subalgebras(&self) -> PyResult<Vec<BranchingRule>> {
        BranchingRule::maximal_subalgebras(self).map_err(PyException::new_err)
    }

    fn __repr__(&self) -> String {
        format!("{}", self)
    }

    fn __str__(&self) -> String {
        format!("{}", self)
    }

    fn __eq__(&self, other: &LieGroup) -> bool {
        return self == other;
    }
}

#[derive(Clone, PartialEq, Hash, Eq)]
#[pyclass]
pub struct SubGroups(pub Vec<LieGroup>);

impl SubGroups {
    pub fn new(algs: Vec<LieGroup>) -> Self {
        Self(
            algs.into_iter()
                .fold(Vec::new(), |mut acc, y| {
                    if !(y == U1 && acc.contains(&U1)) {
                        acc.push(y)
                    }
                    acc
                })
                .tap(|v| v.sort_by(|a, b| b.dim.cmp(&a.dim))),
        )
    }
}

impl Into<SubGroups> for Vec<LieGroup> {
    fn into(self) -> SubGroups {
        SubGroups::new(self)
    }
}

impl Display for SubGroups {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            self.clone()
                .0
                .into_iter()
                .map(|x| format!("{}", x))
                .join("⊗")
        )
    }
}
