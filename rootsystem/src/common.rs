use itertools::Itertools;
use ndarray::{Array, Array2};
use num::rational::Ratio;
use std::{cmp, collections::HashSet, hash::Hash, iter::FromIterator, slice::Iter};

use crate::Array2R;

/// Returns a reflection matrix generated by the vector x;
///
/// # Arguments
/// * `x` - Vector to generate reflection matrix from
pub fn reflection_matrix<'a>(x: &'a Array2R) -> Array2R {
    let n = x.shape()[1];

    Array2R::eye(n)
        - x.t()
            .dot(x)
            .mapv(|y| y * Ratio::from(2) / (x.dot(&x.t())[[0, 0]]))
}

/// Returns whether or not the given weight is dominant or not. Dominant weights
/// are defined by all coefficents being >= 0.
pub fn all_pos<'a>(x: &'a Array2R) -> bool {
    for &i in x.iter() {
        if i < Ratio::from(0) {
            return false;
        }
    }
    true
}

/// Returns whether or not the given weight is dominant or not. Dominant weights
/// are defined by all coefficents being >= 0.
pub fn all_pos_filter<'a>(x: &'a Array2R, filter: Vec<usize>) -> bool {
    for &i in filter.iter() {
        if x[[0, i]] < Ratio::new(0, 1) {
            return false;
        }
    }
    true
}

/// Gets the idx were item is == 0 if gt0 is false,
/// otherwise >= 0
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

/// Returns a unique vec of roots rotated by reflection matrices
///
/// # Arguments
/// * `x` - Vector of vector like arrays that are to be reflected
/// * `reflmats` - Vector of reflection matrices.
pub fn reflect_weights<'a>(x: &'a Vec<Array2R>, reflmats: &'a Vec<Array2R>) -> Vec<Array2R> {
    let mut vecs: Vec<Array2R> = Vec::new();

    for i in x.iter() {
        // Tried `par_iter` here, overhead was slower.
        let reflections: Vec<Array2R> = reflmats.iter().map(|x| i.dot(x)).collect();
        vecs.extend(reflections);
    }
    vecs.extend(x.clone());
    vecs.iter().unique().cloned().collect()
}

pub trait Tap {
    fn tap(self, f: impl FnMut(&mut Self)) -> Self;
}

impl<T> Tap for T {
    fn tap(mut self, mut f: impl FnMut(&mut Self)) -> Self {
        f(&mut self);
        self
    }
}

pub fn set_diff<'a, T>(a: Iter<'a, T>, b: Iter<'a, T>) -> Vec<T>
where
    T: cmp::Eq + Hash + Clone,
{
    let hash_a = HashSet::<T>::from_iter(a.cloned());
    let hash_b = HashSet::<T>::from_iter(b.cloned());

    hash_a.difference(&hash_b).into_iter().cloned().collect()
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

///  Returns a set of unique arrays that are all positive after subtraction by `x`
fn select_pos_diff<'a>(x: &'a Array2R, arrays: &'a Vec<Array2R>) -> HashSet<Array2R> {
    HashSet::from_iter(
        arrays
            .iter()
            .map(|y| x - y)
            .filter(all_pos)
            .clone()
            .into_iter(),
    )
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

/// Returns the unique set of positive arrays being subtracted by `arrays`
pub fn union_new_weights<'a>(
    x: &'a HashSet<Array2R>,
    arrays: &'a Vec<Array2R>,
) -> HashSet<Array2R> {
    let mut res = HashSet::new();
    for w in x.iter() {
        res = res.union(&select_pos_diff(w, arrays)).cloned().collect();
    }
    res.iter().cloned().collect()
}

#[cfg(test)]
pub mod test {
    use super::*;
    use ndarray::{array, Array, Dimension};
    use std::collections::HashSet;
    use std::iter::FromIterator;

    pub fn to_ratio<D>(x: Array<i64, D>) -> Array<Ratio<i64>, D>
    where
        D: Dimension,
    {
        x.mapv(|x| Ratio::new(x, 1))
    }
    #[test]
    fn test_reflection_matrix() {
        let root = to_ratio(array![[1, -1, 0, 0]]);
        let expected = to_ratio(array![
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]);
        assert_eq!(reflection_matrix(&root), expected);
    }

    #[test]
    fn test_select_pos_diff() {
        let weights = vec![
            to_ratio(array![[1, 0, 1]]),
            to_ratio(array![[-1, 1, 1]]),
            to_ratio(array![[1, 1, -1]]),
            to_ratio(array![[-1, 2, -1]]),
        ];

        let result = select_pos_diff(&to_ratio(array![[1, 1, 0]]), &weights);
        let mut expected: HashSet<Array2R> = HashSet::new();
        expected.insert(to_ratio(array![[0, 0, 1]]));
        assert_eq!(result, expected);
    }

    #[test]
    fn test_reflect_weights() {
        let half: num::rational::Ratio<i64> = Ratio::new(1, 2);
        let weights = vec![
            to_ratio(array![[1, -1, 0, 0]]),
            to_ratio(array![[0, 1, -1, 0]]),
        ];

        let reflmats = vec![
            to_ratio(array![
                [0, 1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ]),
            to_ratio(array![
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1]
            ]),
            to_ratio(array![
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1]
            ]),
            array![
                [half, -half, -half, -half],
                [-half, half, -half, -half],
                [-half, -half, half, -half],
                [-half, -half, -half, half],
            ],
        ];

        let result: HashSet<Array2R> =
            HashSet::from_iter(reflect_weights(&weights, &reflmats).into_iter());
        let expected: HashSet<Array2R> = HashSet::from_iter(
            vec![
                to_ratio(array![[-1, 1, 0, 0]]),
                to_ratio(array![[1, 0, -1, 0]]),
                to_ratio(array![[1, -1, 0, 0]]),
                to_ratio(array![[0, -1, 1, 0]]),
                to_ratio(array![[0, 1, 1, 0]]),
                to_ratio(array![[0, 1, -1, 0]]),
            ]
            .into_iter(),
        );

        assert_eq!(result.difference(&expected).count(), 0);
    }
}