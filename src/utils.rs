use ndarray::{Array, Array2, Array3, Ix1};
use num::rational::Ratio;
use numpy::PyReadonlyArray3;
use std::iter::FromIterator;

use crate::debug::Logger;

/// Typedef for rational integer matrix
pub type Array2R = Array2<Ratio<i64>>;

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
            let t = Array::from_iter(i.outer_iter().map(|x| Ratio::new(x[0], x[1])));
            t.into_dimensionality::<Ix1>()
                .ok()
                .unwrap()
                .into_shape((1, ary.shape()[1]))
                .ok()
                .unwrap()
        })
        .collect()
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
    .into_shape((fshape[0] , fshape[1]));
    res.ok().unwrap()
}

#[allow(dead_code)]
pub fn array_to_pyreturn(ary: Array2R) -> (Array2<i64>, Array2<i64>) {
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

#[allow(dead_code)]
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

#[cfg(test)]
mod test {
    use super::*;
    use ndarray::{array, Dimension};
    use num::rational::Ratio;
    use numpy::{get_array_module, PyArray3};
    use pyo3::{prelude::Python, types::IntoPyDict};

    #[allow(dead_code)]
    fn py3darray<'py>(py: Python<'py>) -> &'py PyArray3<i64> {
        py.eval(
            "np.array([[[1, 2], [2, 1], [3, 1]], [[2, 1], [1, 2], [3, 4]]], dtype='int64')",
            Some([("np", get_array_module(py).unwrap())].into_py_dict(py)),
            None,
        )
        .unwrap()
        .downcast()
        .unwrap()
    }

    fn to_ratio<D>(x: Array<i64, D>) -> Array<Ratio<i64>, D>
    where
        D: Dimension,
    {
        x.mapv(|x| Ratio::new(x, 1))
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

    // Tracking bug with cargo test and acquiring gil
    // https://github.com/PyO3/pyo3/issues/771
    #[cfg(not(test))]
    fn test_to_rational_list() {
        pyo3::Python::with_gil(|py| {
            let input = py3darray(py).readonly();
            let result = to_rational_list(input);

            let expected = vec![
                array![[
                    Ratio::new(1i64, 2),
                    Ratio::new(2i64, 1),
                    Ratio::new(3i64, 1)
                ]],
                array![[
                    Ratio::new(2i64, 2),
                    Ratio::new(1i64, 2),
                    Ratio::new(3i64, 4)
                ]],
            ];
            assert_eq!(result, expected);
        })
    }

    #[test]
    fn test_array_to_pyreturn() {
        let input = array![
            [Ratio::new(1i64, 1), Ratio::new(2i64, 1)],
            [Ratio::new(1i64, 2), Ratio::new(0, 1)]
        ];

        let ary1 = array![[1, 2], [1, 0i64]];
        let ary2 = array![[1, 1], [2, 1i64]];

        let result = array_to_pyreturn(input);

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
