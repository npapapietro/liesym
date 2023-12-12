use itertools::Itertools;
use ndarray::{array, s, Array2};
use numpy::{IntoPyArray, PyArray2};
use pyo3::exceptions::PyException;
use pyo3::{pyclass, pymethods, PyResult, Python};
use std::fmt::{self, Display, Formatter};

use crate::utils::find_n_and_m;

use super::{groups::*, AlgResult};
use super::{LieGroup, SubGroups};

macro_rules! subgroups {
    ($($args:expr),*) => {{
        let mut res = Vec::new();
        $(
            res.push($args);
        )*
        SubGroups::new(res)
    }}
}
pub type BranchRuleKey<'a> = &'a str;

#[derive(Clone, PartialEq, Eq, Hash)]
#[pyclass]
pub struct BranchingRule {
    algebra: LieGroup,
    subalgebras: SubGroups,
    /// The string rep of this branching rule
    key: BranchRuleKey<'static>,
}

impl BranchingRule {
    pub fn maximal_subalgebras(algebra: &LieGroup) -> AlgResult<Vec<Self>> {
        let LieGroup { type_, dim } = *algebra;
        match type_ {
            group_types::U1 => Err("Cannot break U1 into subgroups")?,
            group_types::SU => Ok(Su { dim }.branch()?),
            _ => Err("Unsupported algebra")?,
        }
    }

    pub fn projection_matrices(&self) -> AlgResult<Array2<i64>> {
        let LieGroup { dim, type_ } = self.algebra;
        match type_ {
            group_types::SU => {
                Ok(Su { dim }.projection_matrices(self.key, Some(self.subalgebras.clone()))?)
            }
            _ => Err("Unsupported algebra")?,
        }
    }
}

#[pymethods]
impl BranchingRule {
    fn __str__(&self) -> String {
        format!("{}", self)
    }
    fn __repr__(&self) -> String {
        format!("{}", self)
    }
    fn projection_matrix<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArray2<i64>> {
        Ok(self
            .projection_matrices()
            .map_err(PyException::new_err)?
            .into_pyarray(py))
    }
}

impl Display for BranchingRule {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Self {
            subalgebras,
            algebra,
            ..
        } = self;
        write!(f, "{} => {}", algebra, subalgebras)
    }
}

struct Su {
    dim: usize,
}

fn anti_eye(n: usize) -> Array2<i64> {
    let mut m = Array2::zeros((n, n));
    for i in 0..n {
        m[[i, n - i - 1]] = 1;
    }
    m
}
/// Trait that defines the specific implementations per algebra/group for the
/// branching rules.
pub trait BranchingMethods<'a> {
    /**The constant list of branching rules each algebra has.*/
    const BRANCHING_RULES: &'a [BranchRuleKey<'static>];
    /**Branches the algebra into its subalgbras based on branching rules */
    fn branch(&self) -> AlgResult<Vec<BranchingRule>>;

    /// The projecting matrix for given dimension and branching rule. Because some of the keys support multiple
    /// projection matrices, the exact subalgebra on the pattern key can be submitted.
    ///
    fn projection_matrices(
        &self,
        key: BranchRuleKey,
        subalgebra: Option<SubGroups>,
    ) -> AlgResult<Array2<i64>>;
}

impl<'a> BranchingMethods<'a> for Su {
    const BRANCHING_RULES: &'a [BranchRuleKey<'static>] = &[
        "SU(3)=>SU(2)",
        "SU(N+1)=>SU(N)⊗U(1)",
        "SU(N+1)=>SU(K+1)⊗SU(N-K)⊗U(1)",
        "SU(2N+1)=>SO(2N+1)",
        "SU(2N)=>SO(2N)",
        "SU(2N)=>Sp(2N)",
        "SU(MN)=>SU(M)⊗SU(N)",
    ];

    fn branch(&self) -> AlgResult<Vec<BranchingRule>> {
        let mut branches = Vec::new();
        if self.dim == 3 {
            branches.push((
                "SU(3)=>SU(2)",
                subgroups![LieGroup::new("SU", Some(self.dim - 1))?],
            ));
        }

        if self.dim >= 2 {
            branches.push((
                "SU(N+1)=>SU(N)⊗U(1)",
                subgroups![
                    LieGroup::new("SU", Some(self.dim - 1))?,
                    LieGroup::new("U1", None)?
                ],
            ));

            for k in 1..(self.dim - 1) {
                branches.push((
                    "SU(N+1)=>SU(K+1)⊗SU(N-K)⊗U(1)",
                    subgroups![
                        LieGroup::new("SU", Some(k + 1))?,
                        LieGroup::new("SU", Some(self.dim - k - 1))?,
                        LieGroup::new("U1", None)?
                    ],
                ));
            }
        }

        if self.dim % 2 == 1 && self.dim >= 7 {
            branches.push((
                "SU(2N+1)=>SO(2N+1)",
                subgroups![LieGroup::new("SO", Some(self.dim))?],
            ));
        }
        if self.dim >= 8 && self.dim % 2 == 0 {
            branches.push((
                "SU(2N)=>SO(2N)",
                subgroups![LieGroup::new("SO", Some(self.dim))?],
            ));
        }
        if self.dim >= 4 {
            if self.dim % 2 == 0 {
                branches.push((
                    "SU(2N)=>Sp(2N)",
                    subgroups![LieGroup::new("Sp", Some(self.dim))?],
                ));
            }
            for (n, m) in find_n_and_m(self.dim) {
                if n >= 2 && m >= 2 {
                    branches.push((
                        "SU(MN)=>SU(M)⊗SU(N)",
                        subgroups![LieGroup::new("SU", Some(m))?, LieGroup::new("SU", Some(n))?],
                    ));
                }
            }
        }

        Ok(branches
            .into_iter()
            .unique()
            .map(|(key, subalgebras)| BranchingRule {
                algebra: LieGroup {
                    type_: group_types::SU,
                    dim: self.dim,
                },
                subalgebras,
                key,
            })
            .collect())
    }

    // Dev Note: Equations use physics indexing: [1,n] for nxn matrix
    // This means ranging in rust `for i in 0..n` indexes the matrix as `mat[i]`
    // but any equation depending on like `E_i = 2i` must shift to 2 * (i + 1)
    fn projection_matrices(
        &self,
        key: BranchRuleKey,
        subalgebra: Option<SubGroups>,
    ) -> AlgResult<Array2<i64>> {
        let ary = match key {
            "SU(3)=>SU(2)" => array![[2, 2]],
            "SU(N+1)=>SU(N)⊗U(1)" => {
                let n = self.dim - 1;
                let mut mat = Array2::<i64>::zeros((n, n));
                mat.slice_mut(s![..(n - 1), ..(n - 1)])
                    .assign(&Array2::<i64>::eye(n - 1));
                for i in 0..n {
                    mat[[n - 1, i]] = (i + 1) as i64;
                }
                mat
            }
            "SU(N+1)=>SU(K+1)⊗SU(N-K)⊗U(1)" => {
                let sub = subalgebra
                    .ok_or("Subalgebra needed to for key `SU(N+1)=>SU(K+1)⊗SU(N-K)⊗U(1)`")?;

                if sub.0.len() != 3 {
                    return Err("Wrong number of subalgebra for `SU(N+1)=>SU(K+1)⊗SU(N-K)⊗U(1)`");
                }

                // Don't assume order of sub.0
                let k = sub.0.into_iter().find(|x| x.type_ == "SU").unwrap().dim - 1;
                let n = self.dim - 1;

                let mut mat = Array2::<i64>::zeros((n, n));
                mat.slice_mut(s![..k, ..k]).assign(&Array2::<i64>::eye(k));
                mat.slice_mut(s![k..(n - 1), (k + 1)..n])
                    .assign(&Array2::<i64>::eye(n - k - 1));

                for i in 0..k {
                    mat[[n - 1, i]] = ((i + 1) * (n - k)) as i64;
                }
                mat[[n - 1, k]] = ((k + 1) * (n - k)) as i64;

                for i in 0..(n - k) {
                    mat[[n - 1, n - i - 1]] = ((i + 1) * (k + 1)) as i64;
                }

                mat
            }
            "SU(2N+1)=>SO(2N+1)" => {
                let n = self.dim / 2;
                let mut mat = Array2::zeros((n, 2 * n));
                mat.slice_mut(s![..(n - 1), ..(n - 1)])
                    .assign(&Array2::<i64>::eye(n - 1));
                mat.slice_mut(s![..(n - 1), n + 1..2 * n])
                    .assign(&anti_eye(n - 1));
                mat[[n - 1, n - 1]] = 2;
                mat[[n - 1, n]] = 2;
                mat
            }
            "SU(2N)=>Sp(2N)" => {
                let n = self.dim / 2;
                let mut mat = Array2::zeros((n, 2 * n - 1));
                mat.slice_mut(s![..(n - 1), ..(n - 1)])
                    .assign(&Array2::<i64>::eye(n - 1));
                mat.slice_mut(s![..(n - 1), n..2 * n - 1])
                    .assign(&anti_eye(n - 1));
                mat[[n - 1, n - 1]] = 1;
                mat
            }
            "SU(2N)=>SO(2N)" => {
                let n = self.dim / 2;
                let mut mat = Array2::zeros((n, 2 * n - 1));
                mat.slice_mut(s![..(n - 1), ..(n - 1)])
                    .assign(&Array2::<i64>::eye(n - 1));
                mat.slice_mut(s![..(n - 1), n..2 * n - 1])
                    .assign(&anti_eye(n - 1));
                mat[[n - 1, n - 2]] = 1;
                mat[[n - 1, n - 1]] = 2;
                mat[[n - 1, n]] = 1;
                mat
            }
            "SU(MN)=>SU(M)⊗SU(N)" => {
                let sub = subalgebra.ok_or("Subalgebra needed to for key `SU(MN)=>SU(M)⊗SU(N)`")?;

                if sub.0.len() != 2 {
                    return Err("Wrong number of subalgebra for `SU(MN)=>SU(M)⊗SU(N)`");
                }

                let m = sub.0[0].dim;
                let n = sub.0[1].dim;

                let mut mat = Array2::zeros((m + n - 2, m * n - 1));

                // build the 1 2 ... n-1 n n-1 ... 1 vector
                let mut v: Vec<i64> = (1..=n as i64).collect();
                v.extend((1..n as i64).rev());
                let lin = Array2::from_shape_vec((1, v.len()), v).unwrap();

                for j in 0..(m - 1) {
                    let d = 2 * n - 1;
                    mat.slice_mut(s![j..(j + 1), n * j..(n * j + d)])
                        .assign(&lin);
                }

                let max = m + n - 2;
                let mut i = 0;
                while i < m * n - 1 {
                    println!("{}", i);
                    mat.slice_mut(s![(max - n + 1)..max, i * (n - 1)..(i + 1) * (n - 1)])
                        .assign(&Array2::<i64>::eye(n - 1));
                    i += n;
                }

                mat
            }
            _ => return Err("Unknown branching rule"),
        };
        Ok(ary)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_su_projection_matrices() {
        assert_eq!(
            BranchingRule {
                algebra: LieGroup {
                    type_: "SU",
                    dim: 4,
                },
                subalgebras: vec![
                    LieGroup {
                        type_: "SU",
                        dim: 3,
                    },
                    LieGroup {
                        type_: "U1",
                        dim: 1,
                    },
                ]
                .into(),
                key: "SU(N+1)=>SU(N)⊗U(1)",
            }
            .projection_matrices()
            .unwrap(),
            array![[1, 0, 0], [0, 1, 0], [1, 2, 3]]
        );

        assert_eq!(
            BranchingRule {
                algebra: LieGroup {
                    type_: "SU",
                    dim: 5,
                },
                subalgebras: vec![
                    LieGroup {
                        type_: "SU",
                        dim: 3,
                    },
                    LieGroup {
                        type_: "SU",
                        dim: 2,
                    },
                    LieGroup {
                        type_: "U1",
                        dim: 1,
                    },
                ]
                .into(),
                key: "SU(N+1)=>SU(K+1)⊗SU(N-K)⊗U(1)",
            }
            .projection_matrices()
            .unwrap(),
            array![[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [2, 4, 6, 3]]
        );

        assert_eq!(
            BranchingRule {
                algebra: LieGroup {
                    type_: "SU",
                    dim: 9,
                },
                subalgebras: vec![LieGroup {
                    type_: "SO",
                    dim: 9,
                }]
                .into(),
                key: "SU(2N+1)=>SO(2N+1)",
            }
            .projection_matrices()
            .unwrap(),
            array![
                [1, 0, 0, 0, 0, 0, 0, 1],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 1, 0, 0],
                [0, 0, 0, 2, 2, 0, 0, 0],
            ]
        );

        assert_eq!(
            BranchingRule {
                algebra: LieGroup {
                    type_: "SU",
                    dim: 8,
                },
                subalgebras: vec![LieGroup {
                    type_: "Sp",
                    dim: 8,
                }]
                .into(),
                key: "SU(2N)=>Sp(2N)",
            }
            .projection_matrices()
            .unwrap(),
            array![
                [1, 0, 0, 0, 0, 0, 1],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 1, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ]
        );

        assert_eq!(
            BranchingRule {
                algebra: LieGroup {
                    type_: "SU",
                    dim: 8,
                },
                subalgebras: vec![LieGroup {
                    type_: "SO",
                    dim: 8,
                }]
                .into(),
                key: "SU(2N)=>SO(2N)",
            }
            .projection_matrices()
            .unwrap(),
            array![
                [1, 0, 0, 0, 0, 0, 1],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 1, 0, 0],
                [0, 0, 1, 2, 1, 0, 0],
            ]
        );

        assert_eq!(
            BranchingRule {
                algebra: LieGroup {
                    type_: "SU",
                    dim: 10,
                },
                subalgebras: vec![
                    LieGroup {
                        type_: "SU",
                        dim: 5,
                    },
                    LieGroup {
                        type_: "SU",
                        dim: 2,
                    }
                ]
                .into(),
                key: "SU(MN)=>SU(M)⊗SU(N)",
            }
            .projection_matrices()
            .unwrap(),
            array![
                [1, 2, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 2, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 2, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 2, 1],
                [1, 0, 1, 0, 1, 0, 1, 0, 1]
            ]
        );
    }
}
