use ndarray::{Array, Array2, Array3, Ix1};
use num::{rational::Ratio, Complex};
use numpy::{PyReadonlyArray3, PyReadonlyArray4};
use std::ops::FnMut;
use std::{cmp, collections::HashSet, hash::Hash, iter::FromIterator, slice::Iter};

// use crate::debug::Logger;

/// Typedef for rational integer matrix
pub type Array2R = Array2<Ratio<i64>>;
pub type Array2C = Array2<Complex<Ratio<i64>>>;

/// Returns a vector of rational Array2. The first axis divides it into vectors.
///
/// # Arguments
///
/// * `ary` - A 3axis py array of integer type passed in from python. The first two are true
/// axis, while the third is over the p/q of a rational python ratio generated from sympy.
pub fn to_rational_list(ary: PyReadonlyArray3<i64>) -> Vec<Array2R> {
    ary.as_array()
        .outer_iter()
        .map(|i| {
            Array::from_iter(i.outer_iter().map(|x| Ratio::new(x[0], x[1])))
                .into_dimensionality::<Ix1>()
                .ok()
                .unwrap()
                .into_shape((1, ary.shape()[1]))
                .ok()
                .unwrap()
        })
        .collect()
}

/// Utility function for turning incoming Array3 to Array2
fn to_complex_matrix<F>(x: &Array3<i64>, mut f: F) -> Array2C
where
    F: FnMut(i64, i64) -> Complex<Ratio<i64>>,
{
    let shape = x.shape();
    Array::from_iter(
        Array::from_iter(x.iter())
            .into_shape((shape[0] * shape[1], 2))
            .ok()
            .unwrap()
            .outer_iter()
            .map(|a| f(*a[0], *a[1])),
    )
    .into_shape((shape[0], shape[1]))
    .ok()
    .unwrap()
}

/// Converts the incoming matrices from python into list of 2d complex vectors.
///
/// # Arguments
///
/// * `reals` - A 4d matrix with that is X stacked MxN matrices of tupled rationals (X, M, N, 2)
/// * `imags` - A 4d matrix with that is X stacked MxN matrices of tupled rationals (X, M, N, 2)
#[allow(dead_code)]
pub fn to_complex_list(reals: PyReadonlyArray4<i64>, imags: PyReadonlyArray4<i64>) -> Vec<Array2C> {
    reals
        .as_array()
        .outer_iter()
        .zip(imags.as_array().outer_iter())
        .map(|(x, y)| {
            to_complex_matrix(&x.to_owned(), |a, b| Complex::from(Ratio::new(a, b)))
                + to_complex_matrix(&y.to_owned(), |a, b| {
                    Complex::from(Ratio::new(a, b)) * Complex::i()
                })
        })
        .collect::<Vec<_>>()
}

pub fn to_rational_vector(ary: PyReadonlyArray3<i64>) -> Array2R {
    to_rational_list(ary).first().unwrap().clone()
}

/// Returns a matrix of rational Array2.
///
/// # Arguments
///
/// * `ary` - A 3axis py array of integer type passed in from python. The first two are true
/// axis, while the third is over the p/q of a rational python ratio generated from sympy.
pub fn to_rational_matrix(ary: PyReadonlyArray3<i64>) -> Array2R {
    let fshape = &ary.shape();
    let res = Array::from_iter(
        Array::from_iter(ary.as_array().iter())
            .into_shape((fshape[0] * fshape[1], 2))
            .ok()
            .unwrap()
            .outer_iter()
            .map(|x| Ratio::new(*x[0], *x[1])),
    )
    .into_shape((fshape[0], fshape[1]));
    res.ok().unwrap()
}

/// Returns rational array as numer,denom to python
pub fn arrayr_to_pyreturn(ary: Array2R) -> (Array2<i64>, Array2<i64>) {
    (
        ary.mapv(|x| x.numer().clone()),
        ary.mapv(|x| x.denom().clone()),
    )
}

#[allow(dead_code)]
pub fn arrayc_to_pyreturn(ary: Array2R) -> (Array2<i64>, Array2<i64>) {
    (
        ary.mapv(|x| x.numer().clone()),
        ary.mapv(|x| x.denom().clone()),
    )
}

/// Returns the rational array as a tuple of same shaped arrays, (numerator, denominator)
pub fn vecarray_to_pyreturn(ary: Vec<Array2R>) -> (Array3<i64>, Array3<i64>) {
    let n = ary.len();
    let shape = ary[0].shape();

    let mut v: Vec<Ratio<i64>> = Vec::new();
    for i in ary.iter() {
        v.extend(Vec::from_iter(i.iter()));
    }

    let s = Array::from_shape_vec((n, shape[0], shape[1]), v).unwrap();

    (s.mapv(|x| x.numer().clone()), s.mapv(|x| x.denom().clone()))
}

/// Implementation of c++ std::adjacent_find
pub fn adjacent_find(it: Vec<(i64, Array2R)>) -> Vec<usize> {
    let mut v = Vec::new();
    for (idx, i) in it.iter().enumerate() {
        if (idx + 1) >= it.len() {
            break;
        }
        if i.1 == it[idx + 1].1 {
            v.push(idx);
        }
    }
    v
}

pub fn pos_where(ary: Array2R, gt0: bool) -> Vec<usize> {
    ary.iter()
        .enumerate()
        .filter(|(_, &x)| {
            if gt0 {
                x > Ratio::new(0, 1)
            } else {
                x == Ratio::new(0, 1)
            }
        })
        .map(|(i, _)| i)
        .collect()
}

pub fn set_diff<'a, T>(a: Iter<'a, T>, b: Iter<'a, T>) -> Vec<T>
where
    T: cmp::Eq + Hash + Clone,
{
    let hash_a = HashSet::<T>::from_iter(a.cloned());
    let hash_b = HashSet::<T>::from_iter(b.cloned());

    hash_a.difference(&hash_b).into_iter().cloned().collect()
}

pub trait Rational {
    fn to_ratio(&self) -> Array2R;
}

impl Rational for Array2<i64> {
    fn to_ratio(&self) -> Array2R {
        self.mapv(|x| Ratio::new(x, 1))
    }
}

impl Rational for Vec<i64> {
    fn to_ratio(&self) -> Array2R {
        Array::from(self.clone())
            .into_shape((1, self.len()))
            .unwrap()
            .to_ratio()
    }
}

impl Rational for Vec<usize> {
    fn to_ratio(&self) -> Array2R {
        self.into_iter()
            .map(|&x| x as i64)
            .collect::<Vec<_>>()
            .to_ratio()
    }
}

// stolen from https://github.com/rust-lang/rfcs/issues/2178#issuecomment-600368883
pub trait Tap {
    fn tap(self, f: impl FnMut(&mut Self)) -> Self;
}

impl<T> Tap for T {
    fn tap(mut self, mut f: impl FnMut(&mut Self)) -> Self {
        f(&mut self);
        self
    }
}

#[cfg(test)]
pub mod test {
    use super::*;
    use ndarray::{array, Dimension};
    use num::rational::Ratio;
    use numpy::{get_array_module, PyArray3};
    use pyo3::{
        prelude::Python,
        types::{IntoPyDict, PyDict},
    };
    pub fn to_ratio<D: Dimension>(x: Array<i64, D>) -> Array<Ratio<i64>, D> {
        x.mapv(|x| Ratio::new(x, 1))
    }
    fn get_np_locals(py: Python<'_>) -> &'_ PyDict {
        [("np", get_array_module(py).unwrap())].into_py_dict(py)
    }

    // #[allow(dead_code)]
    pub fn py3darray<'py>(py: Python<'py>, mat: String) -> &'py PyArray3<i64> {
        let eval_str = format!("np.array({}, dtype='int64')", mat);
        py.eval(&*eval_str, Some(get_np_locals(py)), None)
            .unwrap()
            .downcast()
            .unwrap()
    }

    #[test]
    fn test_to_ratio() {
        let input = array![[1i64, 2i64], [2i64, 0]];
        let result = to_ratio(input);

        let expected = array![
            [Ratio::new(1i64, 1), Ratio::new(2i64, 1)],
            [Ratio::new(2i64, 1), Ratio::new(0, 1)]
        ];

        assert_eq!(result, expected);
    }

    #[test]
    fn test_to_rational_list() {
        let gil = Python::acquire_gil();
        let py = gil.python();
        get_np_locals(py);
        let mat = "[[[1, 2], [2, 1], [3, 1]], [[2, 1], [1, 2], [3, 4]]]";
        let input = py3darray(py, mat.to_string()).readonly();
        let result = to_rational_list(input);

        let expected = vec![
            array![[
                Ratio::new(1i64, 2),
                Ratio::new(2i64, 1),
                Ratio::new(3i64, 1)
            ]],
            array![[
                Ratio::new(2i64, 1),
                Ratio::new(1i64, 2),
                Ratio::new(3i64, 4)
            ]],
        ];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_to_rational_matrix() {
        let gil = Python::acquire_gil();
        let py = gil.python();
        get_np_locals(py);
        let mat = "
        [
            [[2,1], [-1,1], [0,1]],
            [[-1, 1], [2, 1], [-1,1]],
            [[0,1], [-1,1], [2, 1]]
        ]";
        let input = py3darray(py, mat.to_string()).readonly();
        let result = to_rational_matrix(input);

        let expected = array![
            [
                Ratio::new(2i64, 1),
                Ratio::new(-1i64, 1),
                Ratio::new(0i64, 1)
            ],
            [
                Ratio::new(-1i64, 1),
                Ratio::new(2i64, 1),
                Ratio::new(-1i64, 1)
            ],
            [
                Ratio::new(0i64, 1),
                Ratio::new(-1i64, 1),
                Ratio::new(2i64, 1)
            ]
        ];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_array_to_pyreturn() {
        let input = array![
            [Ratio::new(1i64, 1), Ratio::new(2i64, 1)],
            [Ratio::new(1i64, 2), Ratio::new(0, 1)]
        ];

        let ary1 = array![[1, 2], [1, 0i64]];
        let ary2 = array![[1, 1], [2, 1i64]];

        let result = arrayr_to_pyreturn(input);

        assert_eq!(result, (ary1, ary2));
    }

    #[test]
    fn test_vecarray_to_pyreturn() {
        let input = vec![
            array![
                [Ratio::new(1i64, 1), Ratio::new(2, 1)],
                [Ratio::new(1, 2), Ratio::new(0, 1)]
            ],
            array![
                [Ratio::new(1, 1), Ratio::new(2, 1)],
                [Ratio::new(3, 2), Ratio::new(2, 1)],
            ],
        ];

        let ary1 = array![[[1, 2], [1, 0i64]], [[1, 2], [3, 2i64]]];
        let ary2 = array![[[1, 1], [2, 1i64]], [[1, 1], [2, 1i64]]];

        let result = vecarray_to_pyreturn(input);

        assert_eq!(result, (ary1, ary2));
    }
}
