use itertools::Itertools;
use ndarray::parallel::prelude::IntoParallelIterator;
use ndarray::parallel::prelude::ParallelIterator;
use ndarray::Array;
use num::rational::Ratio;
use numpy::{IntoPyArray, PyArray3, PyReadonlyArray1, PyReadonlyArray3};
use pyo3::prelude::{pyclass, pymethods, Python};
use std::cmp::Ordering;
use std::collections::HashSet;
use std::iter::FromIterator;

use crate::debug::{debug_on, Frac};
use crate::matrix_methods::{
    all_pos, all_pos_filter, reflect_weights, reflection_matrix, union_new_weights,
};
use crate::utils::{
    adjacent_find, to_rational_list, to_rational_matrix, to_rational_vector, vecarray_to_pyreturn,
    Array2R,
};

#[pyclass]
pub struct LieAlgebraBackend {
    rank: usize,
    roots: usize,
    simple_roots: Vec<Array2R>,

    #[allow(dead_code)]
    cartan_matrix: Array2R,
    cartan_matrix_inverse: Array2R,

    omega_matrix: Array2R,
    omega_matrix_inverse: Array2R,

    #[allow(dead_code)]
    cocartan_matrix: Array2R,
}
// Public implementations exposed in python
#[pymethods]
impl LieAlgebraBackend {
    #[new]
    fn new(
        rank: usize,
        roots: usize,
        simple_roots: PyReadonlyArray3<i64>,
        cartan_matrix: PyReadonlyArray3<i64>,
        cartan_matrix_inverse: PyReadonlyArray3<i64>,
        omega_matrix: PyReadonlyArray3<i64>,
        omega_matrix_inverse: PyReadonlyArray3<i64>,
        cocartan_matrix: PyReadonlyArray3<i64>,
    ) -> Self {
        LieAlgebraBackend {
            rank: rank,
            roots: roots,
            simple_roots: to_rational_list(simple_roots),
            cartan_matrix: to_rational_matrix(cartan_matrix),
            cartan_matrix_inverse: to_rational_matrix(cartan_matrix_inverse),
            omega_matrix: to_rational_matrix(omega_matrix),
            omega_matrix_inverse: to_rational_matrix(omega_matrix_inverse),
            cocartan_matrix: to_rational_matrix(cocartan_matrix),
        }
    }

    fn orbit<'py>(
        &self,
        py: Python<'py>,
        weight: PyReadonlyArray3<i64>,
        stabilizers: Option<PyReadonlyArray1<usize>>,
    ) -> (&'py PyArray3<i64>, &'py PyArray3<i64>) {
        let results = match stabilizers {
            Some(x) => self.orbit_stabilizers(
                to_rational_vector(weight),
                x.as_array().iter().map(|i| i.clone()).collect(),
            ),
            None => self.orbit_no_stabilizers(to_rational_vector(weight)),
        };
        let (numer, denom) = vecarray_to_pyreturn(results);
        (numer.into_pyarray(py), denom.into_pyarray(py))
    }

    fn root_system<'py>(&self, py: Python<'py>) -> (&'py PyArray3<i64>, &'py PyArray3<i64>) {
        if debug_on() {
            println!("Made it to root_system")
        }

        let results = self.root_system_backend();
        let (numer, denom) = vecarray_to_pyreturn(results);
        (numer.into_pyarray(py), denom.into_pyarray(py))
    }

    pub fn tensor_product_decomposition<'py>(
        &self,
        py: Python<'py>,
        irrep1: PyReadonlyArray3<i64>,
        irrep2: PyReadonlyArray3<i64>,
    ) -> (&'py PyArray3<i64>, &'py PyArray3<i64>) {
        let results =
            self.tensor_product_decomp(to_rational_vector(irrep1), to_rational_vector(irrep2));
        let (numer, denom) = vecarray_to_pyreturn(results);
        (numer.into_pyarray(py), denom.into_pyarray(py))
    }
}

/// Private implementations not exposed in python
impl LieAlgebraBackend {
    fn get_postive_roots(&self) -> Vec<Array2R> {
        self.root_system_backend()[..(self.roots / 2)].to_vec()
    }

    fn orbit_no_stabilizers(&self, weight: Array2R) -> Vec<Array2R> {
        if debug_on() {
            println!("Starting orbit calculated");
        }
        let mut full_orbit = self.full_orbit(self.to_dominant(weight), None);

        // Sorting by rotating and sum value
        full_orbit.sort_by(|a, b| self.sort_by_ortho(a, b));

        full_orbit
    }

    /// Returns the orbit for weight stabilized around some simple roots.
    ///
    /// # Arguments
    /// * `weight` - The weight to generate the orbit about.
    /// * `stablizers` - A vector of indexes corresponding to the simple roots.
    ///
    fn orbit_stabilizers(&self, weight: Array2R, stablizers: Vec<usize>) -> Vec<Array2R> {
        let mut full_orbit = self.full_orbit(self.to_dominant(weight), Some(stablizers));
        full_orbit.sort_by(|a, b| self.sort_by_ortho(a, b));

        full_orbit
    }

    /// Returns the reflection matrices generated by the simple roots
    fn reflection_matrices(&self) -> Vec<Array2R> {
        self.simple_roots.iter().map(reflection_matrix).collect()
    }

    /// Returns the reflected weights/roots by stabilizers
    ///
    /// # Arguments
    /// * `weights` - Vec of weights to generate reflections about
    /// * `stabilizers` - Optional list of indexes matching simple root generated reflection matrices.
    fn reflect_weights(
        &self,
        weights: Vec<Array2R>,
        stablizers: Option<Vec<usize>>,
    ) -> Vec<Array2R> {
        let reflection_matrices = self.reflection_matrices();
        let ref_mats = match stablizers {
            Some(x) => x.iter().map(|i| reflection_matrices[*i].clone()).collect(),
            None => reflection_matrices,
        };

        reflect_weights(&weights, &ref_mats)
    }

    /// Integer root level for a weight `x`
    fn root_level<'a>(&self, x: &'a Array2R) -> Ratio<i64> {
        x.dot(&self.cartan_matrix_inverse).sum()
    }

    fn ortho_to_omega<'a>(&self, x: &'a Array2R) -> Array2R {
        x.dot(&self.omega_matrix_inverse)
    }

    fn omega_to_ortho(&self, x: Array2R) -> Array2R {
        x.dot(&self.omega_matrix)
    }

    fn omega_to_alpha<'a>(&self, x: &'a Array2R) -> Array2R {
        x.dot(&self.cartan_matrix_inverse)
    }
    /// Returns a list of either count 1 or 0 with the dominant weight
    /// if it exists in the list
    fn find_dom<'a>(&self, arrays: &'a Vec<Array2R>) -> Option<Array2R> {
        for i in arrays.iter() {
            if all_pos(&i.dot(&self.omega_matrix_inverse)) {
                return Some(i.clone());
            }
        }
        None
    }

    /// Returns the dominant weight by rotating across
    /// weyl chambers
    fn to_dominant(&self, weight: Array2R) -> Array2R {
        let mut orbits = vec![weight];
        loop {
            orbits = self.reflect_weights(orbits, None);
            match self.find_dom(&orbits) {
                Some(x) => break x,
                None => continue,
            }
        }
    }

    /// Returns the full orbit with optional stabilization for the weight
    fn full_orbit(&self, weight: Array2R, stablizers: Option<Vec<usize>>) -> Vec<Array2R> {
        let mut orbit = vec![weight];
        for _ in 0..(self.roots / 2) {
            orbit = self.reflect_weights(orbit, stablizers.clone());
        }
        orbit
    }

    /// Calculates all the roots of the algebra in the omega basis.
    /// This is done by generating all the orbits of the simple roots, taking
    /// unique roots in those orbits and ordering them via root level.
    /// Roots of the same level are conventionally ordered by index value,
    /// which is how rust sorts vec![Vec<Integer/Rational>]
    fn root_system_backend(&self) -> Vec<Array2R> {
        let mut roots = self
            .simple_roots
            .clone()
            .into_par_iter()
            .flat_map(|x| self.orbit_no_stabilizers(x))
            .map(|x| self.ortho_to_omega(&x))
            .collect::<Vec<Array2R>>()
            .iter()
            .unique()
            .cloned()
            .collect::<Vec<Array2R>>();

        for _ in 0..self.rank {
            roots.push(Array::zeros((1, self.rank)));
        }
        roots.sort_by(|a, b| self.sort_by_omega(a, b));

        roots
    }

    /// Sorting functor between two roots using omega basis.
    /// Conventions for same level roots to be sorted by convetion
    /// such as (1,1,2) > (1,1,1).
    fn sort_by_omega<'a>(&self, a: &'a Array2R, b: &'a Array2R) -> Ordering {
        let root_level_cmp = self.root_level(b).cmp(&self.root_level(a));
        let convention_ordering = Vec::from_iter(a.iter()).cmp(&Vec::from_iter(b.iter()));

        root_level_cmp.then(convention_ordering)
    }

    /// Sorting functor between two roots using orthogonal basis
    fn sort_by_ortho<'a>(&self, a: &'a Array2R, b: &'a Array2R) -> Ordering {
        let x = self.ortho_to_omega(a);
        let y = self.ortho_to_omega(b);
        self.sort_by_omega(&x, &y)
    }

    /// Returns the single dominant weights of the irreducible representation.
    /// This is done by recursively subtracting positive roots until no new dominant root is
    /// found.
    fn single_dom_weights<'a>(&self, irrep: &'a Array2R) -> Vec<Array2R> {
        let omega_pr: Vec<Array2R> = self.get_postive_roots();

        let mut tower = HashSet::new();
        tower.insert(irrep.clone());
        loop {
            let temp: HashSet<Array2R> = union_new_weights(&tower, &omega_pr);

            let diff = temp.difference(&tower);
            if diff.count() == 0 {
                break;
            }

            tower = tower.union(&temp).cloned().collect();
        }
        tower.into_iter().collect()
    }

    /// Rotate weight or root to a positive chamber
    fn chamber_rotate(&self, weight: Array2R) -> (Array2R, i64) {
        if all_pos(&weight) {
            return (weight, 1);
        }

        let reflection_matrices = self.reflection_matrices();
        let mut reflected = vec![self.omega_to_ortho(weight)];

        let mut num_reflections = 0;
        loop {
            let mut temp = Vec::new();
            for m in reflection_matrices.iter() {
                for r in reflected.iter() {
                    let t = r.dot(m);
                    let ref_omega = self.ortho_to_omega(&t);
                    if all_pos(&ref_omega) {
                        return (ref_omega, (-1i64).pow(num_reflections));
                    }
                    temp.push(t);
                    num_reflections+=1;
                }
            }
            reflected.append(&mut temp)
        }
    }
    fn k_level(&self, x: Array2R) -> Ratio<i64> {
        self.omega_to_alpha(&x).sum()
    }

    /// Scalar product between two weights or roots
    fn scalar_product(&self, a: Array2R, b: Array2R) -> i64 {
        self.omega_to_ortho(a).dot(&self.omega_to_ortho(b).t())[[0, 0]].to_integer()
    }

    /// Returns a list of tuples that are the multiplicty and xi of
    /// each the weight. Xi is defined to be a member of the
    /// positive roots where each coeffcient is positive
    /// nonzero. The multiplicity is the dimension of that
    /// xi's orbit.
    fn xi_multiplicity(&self, weight: Array2R) -> Vec<(Array2R, usize)> {
        let stabs: HashSet<usize> = HashSet::from_iter(
            weight
                .iter()
                .enumerate()
                .filter(|(_, x)| **x == Ratio::new(0, 1))
                .map(|(i, _)| i)
                .into_iter(),
        );
        let mut xi_multiplicity = Vec::new();
        let positive_roots = self.get_postive_roots();
        let xis = positive_roots
            .iter()
            .cloned()
            .filter(|x| all_pos_filter(x, stabs.iter().cloned().collect()));

        for xi in xis {
            let mut xi_stab = HashSet::new();
            let temp = self.omega_to_alpha(&xi);
            for (idx, m) in temp.iter().enumerate() {
                if *m > Ratio::new(0, 1) {
                    xi_stab.insert(idx);
                }
            }

            let diff = xi_stab.difference(&stabs);

            let orbit = self.orbit_stabilizers(
                self.omega_to_ortho(xi.clone()),
                stabs.clone().into_iter().collect(),
            );
            if diff.count() == 0 {
                xi_multiplicity.push((xi, orbit.len()));
            } else {
                xi_multiplicity.push((xi, 2 * orbit.len()));
            }
        }
        return xi_multiplicity;
    }

    pub fn weight_multiplicity_highest_weight(
        &self,
        weight: Array2R,
        irrep: Array2R,
    ) -> Vec<(Array2R, Array2R, usize)> {
        let (dom, _) = self.chamber_rotate(weight.clone());
        let dom_irrep = self.single_dom_weights(&irrep);

        let k = self.k_level(irrep.clone() - weight).to_integer();

        let mut highest_weights = Vec::new();
        for i in 0..k {
            for (xi, mul) in self.xi_multiplicity(dom.clone()).iter() {
                let d = dom.clone() + xi.mapv(|x| x * (i + 1));
                if dom_irrep.contains(&d) {
                    highest_weights.push((d.clone(), xi.clone(), mul.clone()))
                }
            }
        }
        highest_weights
    }

    /// Returns the weight multiplicity in the irreducible representation
    fn weight_multiplicity(&self, weight: Array2R, irrep: Array2R) -> i64 {
        let (dom, _) = self.chamber_rotate(weight.clone());

        if dom == irrep {
            return 1;
        }

        let highest_weights =
            self.weight_multiplicity_highest_weight(weight.clone(), irrep.clone());

        let mut multiplicity = 0;
        let rho = Array2R::ones((1, self.rank));
        // // Freudenthal's Recursion formula
        for (w, xi, m) in highest_weights.iter() {
            let (d, _) = self.chamber_rotate(w.clone());
            let num = self.weight_multiplicity(d.clone(), irrep.clone())
                * self.scalar_product(xi.clone(), xi.clone())
                * (*m as i64);

            let d1 = self.scalar_product(irrep.clone() + rho.clone(), irrep.clone() + rho.clone());
            let d2 = self.scalar_product(dom.clone() + rho.clone(), dom.clone() + rho.clone());

            multiplicity += num / (d1 - d2);
        }

        multiplicity
    }

    fn weight_system(&self, irrep: Array2R) -> Vec<Array2R> {
        let mut dom_weight_system = Vec::new();
        for x in self.single_dom_weights(&irrep).iter() {
            dom_weight_system.push((
                x.clone(),
                self.weight_multiplicity(x.clone(), irrep.clone()),
            ));
        }

        let mut weight_system = Vec::new();

        for (w, _) in dom_weight_system.iter() {
            // for _ in 0..*m {
                let orbit = self.orbit_no_stabilizers(self.omega_to_ortho(w.clone()));
                weight_system.extend(orbit.iter().cloned());
            // }
        }

        weight_system = weight_system
            .iter()
            .map(|v| self.ortho_to_omega(v))
            .collect::<Vec<Array2R>>();

        weight_system.sort_by(|a, b| -> Ordering {
            let k1 = self.k_level(irrep.clone() - a);
            let k2 = self.k_level(irrep.clone() - b);
            k1.cmp(&k2).then( Vec::from_iter(a.iter()).cmp(&Vec::from_iter(b.iter())))
        });

        weight_system
    }

    fn weight_parities(&self, tower: Vec<Array2R>, weight: Array2R) -> Vec<(i64, Array2R)> {
        let rho = Array2R::ones((1, self.rank));

        let mut weight_parities = Vec::new();

        fn any<'a>(x: &'a Array2R) -> bool {
            for &i in x.iter() {
                if i == Ratio::new(0, 1) {
                    return true;
                }
            }
            false
        }

        for w in tower.iter() {
            let (mut t, p) = self.chamber_rotate(w.clone() + weight.clone() + rho.clone());
            if !any(&t) {
                t -= &rho;
                weight_parities.push((p, t.clone()));
            }
        }
        weight_parities
    }

    fn tensor_product_decomp(&self, irrep1: Array2R, irrep2: Array2R) -> Vec<Array2R> {
        let tower = self.weight_system(irrep1);

        let mut weight_parities = self.weight_parities(tower, irrep2.clone());

        weight_parities.sort_by(|a, b| Vec::from_iter(a.1.iter()).cmp(&Vec::from_iter(b.1.iter())));

        for &i in adjacent_find(weight_parities.clone()).iter() {
            weight_parities[i + 1].0 += weight_parities[i].0;
            weight_parities[i].0 = 0;
        }

        let mut tensor_decomp = Vec::new();
        for (m, w) in weight_parities.iter() {
            for _ in 0..*m {
                tensor_decomp.push(w.clone());
            }
        }

        tensor_decomp
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use ndarray::array;
    use std::collections::HashSet;
    use std::iter::FromIterator;

    use crate::utils::test::{py3darray, to_ratio};

    /// Generate A3 algebra
    fn python_strings() -> (usize, usize, String, String, String, String, String, String) {
        let rank = 3;
        let roots = 12;
        let simple_roots_str =
            "[[[1,1],[-1,1],[0,1],[0,1]],[[0,1],[1,1],[-1,1],[0,1]],[[0,1],[0,1],[1,1],[-1,1]]]"
                .to_string();
        let cartan_matrix_str =
            "[[[2,1],[-1,1],[0,1]],[[-1,1],[2,1],[-1,1]],[[0,1],[-1,1],[2,1]]]".to_string();
        let cartan_matrix_inv_str =
            "[[[3,4],[1,2],[1,4]],[[1,2],[1,1],[1,2]],[[1,4],[1,2],[3,4]]]".to_string();
        let omega_matrix_str =
            "[[[3,4],[-1,4],[-1,4],[-1,4]],[[1,2],[1,2],[-1,2],[-1,2]],[[1,4],[1,4],[1,4],[-3,4]]]"
                .to_string();
        let omega_matrix_inv_str =
            "[[[1,1],[0,1],[0,1]],[[-1,1],[1,1],[0,1]],[[0,1],[-1,1],[1,1]],[[0,1],[0,1],[-1,1]]]"
                .to_string();
        let cocartan_matrix_str =
            "[[[1,1],[-1,1],[0,1],[0,1]],[[0,1],[1,1],[-1,1],[0,1]],[[0,1],[0,1],[1,1],[-1,1]]]"
                .to_string();

        (
            rank,
            roots,
            simple_roots_str,
            cartan_matrix_str,
            cartan_matrix_inv_str,
            omega_matrix_str,
            omega_matrix_inv_str,
            cocartan_matrix_str,
        )
    }

    fn helper_liealgebra() -> LieAlgebraBackend {
        let gil = Python::acquire_gil();
        let py = gil.python();

        let (
            rank,
            roots,
            simple_roots_str,
            cartan_matrix_str,
            cartan_matrix_inv_str,
            omega_matrix_str,
            omega_matrix_inv_str,
            cocartan_matrix_str,
        ) = python_strings();

        let simple_roots = py3darray(py, simple_roots_str).readonly();
        let cartan_matrix = py3darray(py, cartan_matrix_str).readonly();
        let cartan_matrix_inv = py3darray(py, cartan_matrix_inv_str).readonly();
        let omega_matrix = py3darray(py, omega_matrix_str).readonly();
        let omega_matrix_inv = py3darray(py, omega_matrix_inv_str).readonly();
        let cocartan_matrix = py3darray(py, cocartan_matrix_str).readonly();
        LieAlgebraBackend::new(
            rank,
            roots,
            simple_roots,
            cartan_matrix,
            cartan_matrix_inv,
            omega_matrix,
            omega_matrix_inv,
            cocartan_matrix,
        )
    }

    #[test]
    fn test_liealgebrabackend_new() {
        // asserts no panics
        helper_liealgebra();
    }

    #[test]
    fn test_to_dom() {
        let algebra = helper_liealgebra();

        let non_dom = to_ratio(array![[1, 0, 1, 0]]);
        let result = algebra.to_dominant(non_dom);

        assert_eq!(result, to_ratio(array![[1, 1, 0, 0]]));
    }

    #[test]
    fn test_root_level() {
        let algebra = helper_liealgebra();
        let tests = vec![
            to_ratio(array![[1, -1, 0, 0]]),
            to_ratio(array![[1, 0, -1, 0]]),
            to_ratio(array![[1, 0, 0, -1]]),
            to_ratio(array![[-1, 1, 0, 0]]),
        ];
        let expected: Vec<i64> = vec![1, 2, 3, -1];

        let results: Vec<i64> = tests
            .iter()
            .map(|x| algebra.root_level(&algebra.ortho_to_omega(&x)).to_integer())
            .collect();

        assert_eq!(results, expected);
    }

    #[test]
    fn test_ortho_to_omega() {
        let algebra = helper_liealgebra();
        let tests = vec![
            to_ratio(array![[1, -1, 0, 0]]),
            to_ratio(array![[1, 0, -1, 0]]),
            to_ratio(array![[1, 0, 0, -1]]),
            to_ratio(array![[-1, 1, 0, 0]]),
        ];
        let expected = vec![
            to_ratio(array![[2, -1, 0]]),
            to_ratio(array![[1, 1, -1]]),
            to_ratio(array![[1, 0, 1]]),
            to_ratio(array![[-2, 1, 0]]),
        ];

        let results: Vec<Array2R> = tests.iter().map(|x| algebra.ortho_to_omega(&x)).collect();

        assert_eq!(results, expected);
    }

    #[test]
    fn test_full_orbit() {
        let algebra = helper_liealgebra();

        let non_dom = to_ratio(array![[1, 0, 1, 0]]);
        let result: HashSet<Array2R> =
            HashSet::from_iter(algebra.full_orbit(non_dom, None).into_iter());

        let expected: HashSet<Array2R> = HashSet::from_iter(
            vec![
                to_ratio(array![[1, 0, 1, 0]]),
                to_ratio(array![[1, 0, 0, 1]]),
                to_ratio(array![[1, 1, 0, 0]]),
                to_ratio(array![[0, 1, 1, 0]]),
                to_ratio(array![[0, 1, 0, 1]]),
                to_ratio(array![[0, 0, 1, 1]]),
            ]
            .into_iter(),
        );
        // order doesn't matter
        assert_eq!(result, expected);
    }

    #[test]
    fn test_orbit_no_stabilizers() {
        let algebra = helper_liealgebra();
        let sr = to_ratio(array![[1, -1, 0, 0]]);

        let results: HashSet<Array2R> =
            HashSet::from_iter(algebra.orbit_no_stabilizers(sr).into_iter());
        let expected: HashSet<Array2R> = HashSet::from_iter(
            vec![
                to_ratio(array![[-1, 0, 0, 1]]),
                to_ratio(array![[-1, 0, 1, 0]]),
                to_ratio(array![[-1, 1, 0, 0]]),
                to_ratio(array![[0, -1, 0, 1]]),
                to_ratio(array![[0, -1, 1, 0]]),
                to_ratio(array![[0, 0, -1, 1]]),
                to_ratio(array![[0, 0, 1, -1]]),
                to_ratio(array![[0, 1, -1, 0]]),
                to_ratio(array![[0, 1, 0, -1]]),
                to_ratio(array![[1, -1, 0, 0]]),
                to_ratio(array![[1, 0, -1, 0]]),
                to_ratio(array![[1, 0, 0, -1]]),
            ]
            .into_iter(),
        );

        assert_eq!(results, expected);
    }

    #[test]
    fn test_rootsystem_backend() {
        let algebra = helper_liealgebra();
        let results = algebra.root_system_backend();
        let expected = vec![
            to_ratio(array![[1, 0, 1]]),
            to_ratio(array![[-1, 1, 1]]),
            to_ratio(array![[1, 1, -1]]),
            to_ratio(array![[-1, 2, -1]]),
            to_ratio(array![[0, -1, 2]]),
            to_ratio(array![[2, -1, 0]]),
            to_ratio(array![[0, 0, 0]]),
            to_ratio(array![[0, 0, 0]]),
            to_ratio(array![[0, 0, 0]]),
            to_ratio(array![[-2, 1, 0]]),
            to_ratio(array![[0, 1, -2]]),
            to_ratio(array![[1, -2, 1]]),
            to_ratio(array![[-1, -1, 1]]),
            to_ratio(array![[1, -1, -1]]),
            to_ratio(array![[-1, 0, -1]]),
        ];
        assert_eq!(results, expected)
    }

    #[test]
    fn test_weight_system() {
        let algebra = helper_liealgebra();
        let results = algebra.weight_system(to_ratio(array![[1, 0, 0]]));
        let expected = vec![
            to_ratio(array![[1, 0, 0]]),
            to_ratio(array![[-1, 1, 0]]),
            to_ratio(array![[0, -1, 1]]),
            to_ratio(array![[0, 0, -1]]),
        ];
        assert_eq!(results, expected);

        let results2 = algebra.weight_system(to_ratio(array![[2, 0, 0]]));
        let expected2 = vec![
            to_ratio(array![[2, 0, 0]]),
            to_ratio(array![[0, 1, 0]]),
            to_ratio(array![[-2, 2, 0]]),
            to_ratio(array![[1, -1, 1]]),
            to_ratio(array![[-1, 0, 1]]),
            to_ratio(array![[1, 0, -1]]),
            to_ratio(array![[-1, 1, -1]]),
            to_ratio(array![[0, -2, 2]]),
            to_ratio(array![[0, -1, 0]]),
            to_ratio(array![[0, 0, -2]]),
        ];

        assert_eq!(results2, expected2);
    }

    #[test]
    fn test_xi_multiplicity() {
        let algebra = helper_liealgebra();
        let results = algebra.xi_multiplicity(to_ratio(array![[0, 1, 0]]));
        assert_eq!(
            results,
            vec![
                (to_ratio(array![[1, 0, 1]]), 8),
                (to_ratio(array![[0, -1, 2]]), 4),
                (to_ratio(array![[2, -1, 0]]), 4),
            ]
        )
    }

    #[test]
    fn test_weight_multiplicity_highest_weight() {
        let algebra = helper_liealgebra();
        let result = algebra.weight_multiplicity_highest_weight(
            to_ratio(array![[0, 0, 2]]),
            to_ratio(array![[1, 1, 1]]),
        );
        let expected: Vec<(Array2R, Array2R, usize)> =
            vec![(to_ratio(array![[1, 1, 1]]), to_ratio(array![[1, 1, -1]]), 3)];

        assert_eq!(result, expected)
    }



    #[test]
    fn test_single_dom_weights() {
        let algebra = helper_liealgebra();

        {
            let result: HashSet<Array2R> = HashSet::from_iter(
                algebra
                    .single_dom_weights(&to_ratio(array![[1, 1, 1]]))
                    .into_iter(),
            );
    
            let expected: HashSet<Array2R> = HashSet::from_iter(
                vec![
                    to_ratio(array![[1, 1, 1]]),
                    to_ratio(array![[2, 0, 0]]),
                    to_ratio(array![[0, 0, 2]]),
                    to_ratio(array![[0, 1, 0]]),
                ]
                .into_iter(),
            );
    
            assert_eq!(result, expected)
        }

        {
            let result: HashSet<Array2R> = HashSet::from_iter(
                algebra
                    .single_dom_weights(&to_ratio(array![[2, 0, 0]]))
                    .into_iter(),
            );
    
            let expected: HashSet<Array2R> = HashSet::from_iter(
                vec![
                    to_ratio(array![[0, 1, 0]]),
                    to_ratio(array![[2, 0, 0]]),
     
                ]
                .into_iter(),
            );
    
            assert_eq!(result, expected)
        }
        
    }

    #[test]
    fn test_positive_roots() {
        let algebra = helper_liealgebra();
        let result = algebra.get_postive_roots();
        let expected = vec![
            to_ratio(array![[1, 0, 1]]),
            to_ratio(array![[-1, 1, 1]]),
            to_ratio(array![[1, 1, -1]]),
            to_ratio(array![[-1, 2, -1]]),
            to_ratio(array![[0, -1, 2]]),
            to_ratio(array![[2, -1, 0]]),
        ];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_weight_multiplicity() {
        let algebra = helper_liealgebra();

        let result =
            algebra.weight_multiplicity(to_ratio(array![[0, 0, 2]]), to_ratio(array![[1, 1, 1]]));
        assert_eq!(result, 1)
    }

    #[test]
    fn test_weight_parities() {
        let algebra = helper_liealgebra();
        let tower = algebra.weight_system(to_ratio(array![[1, 0, 0]]));
        let results: HashSet<_> = HashSet::from_iter(
            algebra
                .weight_parities(tower, to_ratio(array![[0, 0, 1]]))
                .into_iter(),
        );

        let expected: HashSet<_> = HashSet::from_iter(
            vec![
                (1, to_ratio(array![[1, 0, 1]])),
                (1, to_ratio(array![[0, 0, 0]])),
            ]
            .into_iter(),
        );

        assert_eq!(results, expected)
    }

    #[test]
    fn test_tensorproduct() {
        let algebra = helper_liealgebra();

        let decomp =
            algebra.tensor_product_decomp(to_ratio(array![[1, 0, 0]]), to_ratio(array![[1, 0, 0]]));
        let results: HashSet<_> = HashSet::from_iter(decomp.clone().into_iter());
        let expected: HashSet<_> = HashSet::from_iter(
            vec![to_ratio(array![[2, 0, 0]]), to_ratio(array![[0, 1, 0]])].into_iter(),
        );

        assert_eq!(results, expected, "Two term tensordecmop is not correct");

        let mut n_results = Vec::new();
        for r in decomp.iter() {
            n_results.extend(algebra.tensor_product_decomp(r.clone(), to_ratio(array![[1, 0, 0]])))
        }

        assert_eq!(n_results.len(), 4, "Three term tensordecomp is not correct");
    }

    // #[test]
    // fn test_adsad(){
    //     let irrep1 = to_ratio(array![[2, 0, 0]]);
    //     let irrep2 = to_ratio(array![[1, 0, 0]]);
    //     let algebra = helper_liealgebra();
    //     let tower = algebra.weight_system(irrep1);

    //     let weight_parities = algebra.weight_parities(tower, irrep2.clone());
    //     for (idx, w) in weight_parities.iter(){
    //         println!("Weight: {:?}, mul: {:?}", Frac::format(w.clone()), idx.clone());
    //     }
    // }
}
