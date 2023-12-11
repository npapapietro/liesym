use itertools::Itertools;
use ndarray::parallel::prelude::{IntoParallelIterator, ParallelIterator};
use num::rational::Ratio;
use std::cmp::Ordering;
use std::collections::HashSet;
use std::iter::FromIterator;

use crate::common::{
    adjacent_find, all_pos, all_pos_filter, pos_where, reflect_weights, reflection_matrix,
    set_diff, union_new_weights, Tap,
};
use crate::Array2R;

#[derive(Debug)]
pub struct RootSystem {
    pub rank: usize,
    pub n_roots: usize,
    pub simple_roots: Vec<Array2R>,
    pub cartan_matrix_inverse: Array2R,
    pub omega_matrix: Array2R,
    pub omega_matrix_inverse: Array2R,
}

impl RootSystem {
    pub fn new(
        rank: usize,
        n_roots: usize,
        simple_roots: Vec<Array2R>,
        cartan_matrix_inverse: Array2R,
        omega_matrix: Array2R,
        omega_matrix_inverse: Array2R,
    ) -> Self {
        Self {
            rank,
            n_roots,
            simple_roots,
            cartan_matrix_inverse,
            omega_matrix,
            omega_matrix_inverse,
        }
    }

    /// Generates the root/weight level by adding the
    /// dynkin coefficients up.
    fn root_level<'a>(&self, x: &'a Array2R) -> Ratio<i64> {
        self.omega_to_alpha(x).sum()
    }

    fn ortho_to_omega<'a>(&self, x: &'a Array2R) -> Array2R {
        x.dot(&self.omega_matrix_inverse)
    }

    fn omega_to_ortho(&self, x: &Array2R) -> Array2R {
        x.dot(&self.omega_matrix)
    }

    fn omega_to_alpha<'a>(&self, x: &'a Array2R) -> Array2R {
        x.dot(&self.cartan_matrix_inverse)
    }

    /// Generates the roots reflection matrices
    fn reflection_matrices(&self) -> Vec<Array2R> {
        self.simple_roots.iter().map(reflection_matrix).collect()
    }

    /// Returns a list of either count 1 or 0 with the dominant weight
    /// if it exists in the list
    fn find_dom<'a>(&self, arrays: &'a Vec<Array2R>) -> Option<Array2R> {
        for i in arrays.iter() {
            if all_pos(&self.omega_to_alpha(&self.ortho_to_omega(i))) {
                return Some(i.clone());
            }
        }
        None
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
            Some(x) => x.iter().map(|&i| reflection_matrices[i].clone()).collect(),
            None => reflection_matrices,
        };

        reflect_weights(&weights, &ref_mats)
    }

    /// Scalar product between two weights or roots
    /// # Arguments
    /// * `a` - Weight in omega basis
    /// * `b` - Weight in omega basis
    pub fn scalar_product(&self, a: &Array2R, b: &Array2R) -> Ratio<i64> {
        self.omega_to_ortho(a).dot(&self.omega_to_ortho(b).t())[[0, 0]]
    }

    pub fn index_irrep<'a>(&self, irrep: &'a Array2R, dim: i64) -> Ratio<i64> {
        let delta = &Array2R::ones((1, self.rank));
        self.scalar_product(irrep, &(irrep + delta * 2)) * dim / (self.n_roots as i64)
    }

    /// Sorting functor between two roots using orthogonal basis
    fn sort_by_ortho<'a>(&self, a: &'a Array2R, b: &'a Array2R) -> Ordering {
        let x = self.ortho_to_omega(a);
        let y = self.ortho_to_omega(b);
        self.sort_by_omega(&x, &y)
    }

    /// Sorting functor between two roots using omega basis.
    /// Conventions for same level roots to be sorted by convetion
    /// such as (1,1,2) > (1,1,1).
    fn sort_by_omega<'a>(&self, a: &'a Array2R, b: &'a Array2R) -> Ordering {
        let root_level_cmp = self.root_level(b).cmp(&self.root_level(a));
        let convention_ordering = Vec::from_iter(a.iter()).cmp(&Vec::from_iter(b.iter()));

        root_level_cmp.then(convention_ordering)
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
        for _ in 0..self.n_roots {
            orbit = self.reflect_weights(orbit, stablizers.clone());
        }
        orbit
    }

    /// Calculates all the roots of the algebra in the omega basis.
    /// This is done by generating all the orbits of the simple roots, taking
    /// unique roots in those orbits and ordering them via root level.
    /// Roots of the same level are conventionally ordered by index value,
    /// which is how rust sorts vec![Vec<Integer/Rational>]
    ///
    /// Returns in the omega basis
    pub fn root_system(&self) -> Vec<Array2R> {
        let mut roots = self
            .simple_roots
            .clone()
            .into_par_iter()
            .flat_map(|x| self.orbit_no_stabilizers(x))
            .map(|x| {
                let x = self.ortho_to_omega(&x);
                x
            })
            .collect::<Vec<Array2R>>()
            .iter()
            .unique()
            .cloned()
            .collect::<Vec<Array2R>>();

        for _ in 0..self.rank {
            roots.push(Array2R::zeros((1, self.rank)));
        }
        roots.sort_by(|a, b| self.sort_by_omega(a, b));

        roots
    }

    /// Returns the orbit for weight stabilized around some simple roots.
    /// Returned basis is orthogonal.
    ///
    /// # Arguments
    /// * `weight` - The weight to generate the orbit about.
    /// * `stablizers` - A vector of indexes corresponding to the simple roots.
    ///
    fn orbit_stabilizers(&self, weight: Array2R, stablizers: Vec<usize>) -> Vec<Array2R> {
        self.full_orbit(self.to_dominant(weight), Some(stablizers))
            .tap(|x| x.sort_by(|a, b| self.sort_by_ortho(a, b)))
    }

    fn orbit_no_stabilizers(&self, weight: Array2R) -> Vec<Array2R> {
        self.full_orbit(self.to_dominant(weight), None).tap(|x|
            // Sorting by rotating and sum value
        x.sort_by(|a, b| self.sort_by_ortho(a, b)))
    }

    pub fn orbit(&self, weight: Array2R, stabs: Option<Vec<usize>>) -> Vec<Array2R> {
        match stabs {
            Some(x) => self.orbit_stabilizers(weight, x),
            None => self.orbit_no_stabilizers(weight),
        }
    }

    pub fn get_postive_roots(&self) -> Vec<Array2R> {
        self.root_system()[..self.n_roots].to_vec()
    }

    pub fn dim(&self, irrep: &Array2R) -> i64 {
        let rho = &Array2R::ones((1, self.rank));

        let mut dim = Ratio::from(1);

        for root in self.get_postive_roots().iter() {
            dim *= self.scalar_product(&(irrep + rho), root) / self.scalar_product(rho, root);
        }

        dim.to_integer()
    }

    fn xis<'a>(&self, stabs: &'a Vec<usize>) -> Vec<Array2R> {
        return self
            .get_postive_roots()
            .iter()
            .cloned()
            .filter(|x| all_pos_filter(x, stabs.clone()))
            .collect();
    }

    /// Returns a list of tuples that are the multiplicty and xi of
    /// each the weight. Xi is defined to be a member of the
    /// positive roots where each coeffcient is positive
    /// nonzero. The multiplicity is the dimension of that
    /// xi's orbit.
    fn xi_multiplicity(&self, weight: Array2R) -> Vec<(Array2R, usize)> {
        let stabs: Vec<usize> = pos_where(weight.clone(), false);

        let mut xi_multiplicity = Vec::new();

        let xis = self.xis(&stabs.clone());

        for xi in xis {
            let w = self.omega_to_ortho(&xi);

            let orbit = self.full_orbit(w.clone(), Some(stabs.clone()));

            let diff = set_diff(
                pos_where(self.omega_to_alpha(&xi).clone(), true).iter(),
                stabs.iter(),
            );
            if diff.len() == 0 {
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
        let (dom, _) = self.reflect_to_dominant(weight.clone(), None);
        let dom_irrep = self.single_dom_weights(&irrep);

        let k = self.root_level(&(irrep.clone() - weight)).to_integer();

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
    fn weight_multiplicity(&self, weight: &Array2R, irrep: &Array2R) -> i64 {
        let (dom, _) = self.reflect_to_dominant(weight.clone(), None);

        if dom == irrep {
            return 1;
        }

        let highest_weights =
            self.weight_multiplicity_highest_weight(weight.clone(), irrep.clone());

        let mut multiplicity = Ratio::new(0, 1);
        let rho = Array2R::ones((1, self.rank));

        // // Freudenthal's Recursion formula
        for (w, xi, m) in highest_weights.iter() {
            let (d, _) = self.reflect_to_dominant(w.clone(), None);

            let num = self.scalar_product(w, xi).clone()
                * self.weight_multiplicity(&d, irrep)
                * (*m as i64);

            let i_r = irrep + &rho;
            let d_r = &dom + &rho;
            let d1 = self.scalar_product(&i_r, &i_r);
            let d2 = self.scalar_product(&d_r, &d_r);

            multiplicity += num / (d1 - d2);
        }

        multiplicity.to_integer()
    }

    fn weight_system_with_mul(&self, irrep: Array2R) -> Vec<(Array2R, i64)> {
        let dom_weight_system: Vec<_> = self
            .single_dom_weights(&irrep)
            .iter()
            .map(|x| (x.clone(), self.weight_multiplicity(x, &irrep)))
            .collect();

        dom_weight_system
            .iter()
            .flat_map(|(w, m)| {
                let ortho = self.omega_to_ortho(w);
                let result: Vec<_> = self
                    .orbit_no_stabilizers(ortho)
                    .iter()
                    .map(|x| (x.clone(), m.clone()))
                    .collect();
                result
            })
            .map(|(w, m)| (self.ortho_to_omega(&w), m))
            .collect::<Vec<_>>()
            .tap(|x| {
                x.sort_by(|a, b| -> Ordering {
                    let k1 = self.root_level(&(irrep.clone() - a.clone().0));
                    let k2 = self.root_level(&(irrep.clone() - b.clone().0));
                    k1.cmp(&k2)
                        .then(Vec::from_iter(a.0.iter()).cmp(&Vec::from_iter(b.0.iter())))
                })
            })
    }

    fn weight_parities(
        &self,
        tower_with_mul: Vec<(Array2R, i64)>,
        weight: Array2R,
    ) -> Vec<(i64, Array2R)> {
        let rho = Array2R::ones((1, self.rank));

        tower_with_mul
            .iter()
            .map(|(w, m)| {
                self.reflect_to_dominant(w.clone() + weight.clone() + rho.clone(), Some(m.clone()))
            })
            .filter(|(t, _)| !t.iter().any(|&x| x == Ratio::new(0, 1)))
            .map(|(x, y)| (y, x - &rho))
            .collect()
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

    /// Rotate weight or root to a positive (dominant) chamber.
    /// # Arguments
    /// * `weight` - Weight in omega basis
    fn reflect_to_dominant(&self, weight: Array2R, mul: Option<i64>) -> (Array2R, i64) {
        if all_pos(&weight) {
            return (weight, 1);
        }

        let reflection_matrices = self.reflection_matrices();
        let mut reflected = vec![self.omega_to_ortho(&weight)];

        let mut num_reflections = 0;
        loop {
            let mut temp = Vec::new();
            for m in reflection_matrices.iter() {
                for r in reflected.iter() {
                    // algorithm counts before the reflections
                    num_reflections += 1;

                    let t = r.dot(m);
                    let ref_omega = self.ortho_to_omega(&t);
                    if all_pos(&ref_omega) {
                        return (
                            ref_omega,
                            mul.unwrap_or(1i64) * (-1i64).pow(num_reflections),
                        );
                    }
                    temp.push(t);
                }
            }
            reflected.append(&mut temp)
        }
    }

    pub fn tensor_product_decomp(&self, irrep1: Array2R, irrep2: Array2R) -> Vec<Array2R> {
        let tower = self.weight_system_with_mul(irrep1);

        let mut weight_parities = self
            .weight_parities(tower, irrep2.clone())
            .tap(|x| x.sort_by(|a, b| Vec::from_iter(a.1.iter()).cmp(&Vec::from_iter(b.1.iter()))));

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
pub mod test {
    use super::*;
    use ndarray::array;
    use std::collections::HashSet;
    use std::iter::FromIterator;

    use crate::common::test::to_ratio;

    enum GroupTestType {
        A1,
        A,
        B,
    }

    fn helper_liealgebra(group_type: GroupTestType) -> RootSystem {
        match group_type {
            GroupTestType::A1 => RootSystem::new(
                1,
                1,
                vec![to_ratio(array![[1, -1]])],
                array![[Ratio::new(1, 2)]],
                array![[Ratio::new(1, 2), Ratio::new(-1, 2)]],
                array![[Ratio::from(1)], [Ratio::from(-1)]],
            ),
            GroupTestType::A => RootSystem::new(
                3,
                6,
                vec![
                    to_ratio(array![[1, -1, 0, 0]]),
                    to_ratio(array![[0, 1, -1, 0]]),
                    to_ratio(array![[0, 0, 1, -1]]),
                ],
                array![
                    [Ratio::new(3, 4), Ratio::new(1, 2), Ratio::new(1, 4)],
                    [Ratio::new(1, 2), Ratio::new(1, 1), Ratio::new(1, 2)],
                    [Ratio::new(1, 4), Ratio::new(1, 2), Ratio::new(3, 4)]
                ],
                array![
                    [
                        Ratio::new(3, 4),
                        Ratio::new(-1, 4),
                        Ratio::new(-1, 4),
                        Ratio::new(-1, 4)
                    ],
                    [
                        Ratio::new(1, 2),
                        Ratio::new(1, 2),
                        Ratio::new(-1, 2),
                        Ratio::new(-1, 2)
                    ],
                    [
                        Ratio::new(1, 4),
                        Ratio::new(1, 4),
                        Ratio::new(1, 4),
                        Ratio::new(-3, 4)
                    ]
                ],
                array![
                    [Ratio::new(1, 1), Ratio::new(0, 1), Ratio::new(0, 1)],
                    [Ratio::new(-1, 1), Ratio::new(1, 1), Ratio::new(0, 1)],
                    [Ratio::new(0, 1), Ratio::new(-1, 1), Ratio::new(1, 1)],
                    [Ratio::new(0, 1), Ratio::new(0, 1), Ratio::new(-1, 1)]
                ],
            ),
            GroupTestType::B => RootSystem::new(
                3,
                9,
                vec![
                    to_ratio(array![[1, -1, 0]]),
                    to_ratio(array![[0, 1, -1]]),
                    to_ratio(array![[0, 0, 1,]]),
                ],
                array![
                    [Ratio::new(1, 1), Ratio::new(1, 1), Ratio::new(1, 1)],
                    [Ratio::new(1, 1), Ratio::new(2, 1), Ratio::new(2, 1)],
                    [Ratio::new(1, 2), Ratio::new(1, 1), Ratio::new(3, 2)]
                ],
                array![
                    [Ratio::new(1, 1), Ratio::new(0, 1), Ratio::new(0, 1)],
                    [Ratio::new(1, 1), Ratio::new(1, 1), Ratio::new(0, 1)],
                    [Ratio::new(1, 2), Ratio::new(1, 2), Ratio::new(1, 2)]
                ],
                array![
                    [Ratio::new(1, 1), Ratio::new(0, 1), Ratio::new(0, 1)],
                    [Ratio::new(-1, 1), Ratio::new(1, 1), Ratio::new(0, 1)],
                    [Ratio::new(0, 1), Ratio::new(-1, 1), Ratio::new(2, 1)]
                ],
            ),
        }
    }

    #[test]
    fn test_liealgebrabackend_new() {
        // asserts no panics
        helper_liealgebra(GroupTestType::A);

        helper_liealgebra(GroupTestType::B);
    }

    #[test]
    fn test_to_dom() {
        {
            let algebra = helper_liealgebra(GroupTestType::A);

            let non_dom = to_ratio(array![[1, 0, 1, 0]]);
            let result = algebra.to_dominant(non_dom);

            assert_eq!(result, to_ratio(array![[1, 1, 0, 0]]));
        }
        // {
        //     let algebra = helper_liealgebra(GroupTestType::B);

        //     let non_dom = array![[
        //         Ratio::<i64>::new(1, 2),
        //         Ratio::<i64>::new(-1, 2),
        //         Ratio::<i64>::new(-1, 2)
        //     ]];
        //     let result = algebra.to_dominant(non_dom);

        //     assert_eq!(
        //         result,
        //         array![[
        //             Ratio::<i64>::new(1, 2),
        //             Ratio::<i64>::new(1, 2),
        //             Ratio::<i64>::new(1, 2)
        //         ]]
        //     );
        // }
    }

    #[test]
    fn test_root_level() {
        let algebra = helper_liealgebra(GroupTestType::A);
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
    fn test_reflect_weights() {
        {
            let algebra = helper_liealgebra(GroupTestType::A);
            let stabs = vec![0, 1];
            let root = to_ratio(array![[1, 0, 0, 0]]);

            let results = algebra.reflect_weights(vec![root], Option::Some(stabs));

            assert_eq!(
                results,
                vec![
                    to_ratio(array![[0, 1, 0, 0]]),
                    to_ratio(array![[1, 0, 0, 0]])
                ],
                "Group A reflect_weights error"
            );
        }

        {
            let algebra = helper_liealgebra(GroupTestType::B);
            let stabs = vec![0, 1];
            let root = to_ratio(array![[1, 0, 0]]);

            let results = algebra.reflect_weights(vec![root], Option::Some(stabs));

            assert_eq!(
                results,
                vec![to_ratio(array![[0, 1, 0]]), to_ratio(array![[1, 0, 0]])],
                "Group B reflect_weights error"
            );
        }
    }

    #[test]
    fn test_ortho_to_omega() {
        let algebra = helper_liealgebra(GroupTestType::A);
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
        {
            let algebra = helper_liealgebra(GroupTestType::A);

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
            assert_eq!(result, expected, "Type A error");
        }
        {
            let algebra = helper_liealgebra(GroupTestType::B);
            let non_dom = to_ratio(array![[0, 1, 0]]);
            let result: HashSet<Array2R> =
                HashSet::from_iter(algebra.full_orbit(non_dom, Option::Some(vec![1, 2])));

            let expected: HashSet<Array2R> = HashSet::from_iter(
                vec![
                    to_ratio(array![[0, 1, 0]]),
                    to_ratio(array![[0, 0, 1]]),
                    to_ratio(array![[0, 0, -1]]),
                    to_ratio(array![[0, -1, 0]]),
                ]
                .into_iter(),
            );

            assert_eq!(result, expected, "Type B error")
        }
    }

    #[test]
    fn test_orbit_no_stabilizers() {
        let algebra = helper_liealgebra(GroupTestType::A);
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
        {
            let algebra = helper_liealgebra(GroupTestType::A);
            let results = algebra.root_system();
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
        {
            let algebra = helper_liealgebra(GroupTestType::A1);
            let results = algebra.root_system();
            let expected = vec![
                to_ratio(array![[2]]),
                to_ratio(array![[0]]),
                to_ratio(array![[-2]]),
            ];
            assert_eq!(results, expected)
        }
    }

    #[test]
    fn test_weight_system_with_mul() {
        {
            let algebra = helper_liealgebra(GroupTestType::A);
            let results = algebra.weight_system_with_mul(to_ratio(array![[1, 0, 0]]));
            let expected = vec![
                (to_ratio(array![[1, 0, 0]]), 1),
                (to_ratio(array![[-1, 1, 0]]), 1),
                (to_ratio(array![[0, -1, 1]]), 1),
                (to_ratio(array![[0, 0, -1]]), 1),
            ];
            assert_eq!(set_diff(results.iter(), expected.iter()).len(), 0);

            let results2 = algebra.weight_system_with_mul(to_ratio(array![[2, 0, 0]]));
            let expected2 = vec![
                (to_ratio(array![[2, 0, 0]]), 1),
                (to_ratio(array![[0, 1, 0]]), 1),
                (to_ratio(array![[-2, 2, 0]]), 1),
                (to_ratio(array![[1, -1, 1]]), 1),
                (to_ratio(array![[-1, 0, 1]]), 1),
                (to_ratio(array![[1, 0, -1]]), 1),
                (to_ratio(array![[-1, 1, -1]]), 1),
                (to_ratio(array![[0, -2, 2]]), 1),
                (to_ratio(array![[0, -1, 0]]), 1),
                (to_ratio(array![[0, 0, -2]]), 1),
            ];

            assert_eq!(set_diff(results2.iter(), expected2.iter()).len(), 0);
        }
        {
            let algebra = helper_liealgebra(GroupTestType::B);
            let results = algebra.weight_system_with_mul(to_ratio(array![[2, 0, 0]]));

            let expected = vec![
                (to_ratio(array![[2, 0, 0]]), 1),
                (to_ratio(array![[-2, 2, 0]]), 1),
                (to_ratio(array![[0, -2, 4]]), 1),
                (to_ratio(array![[0, 2, -4]]), 1),
                (to_ratio(array![[2, -2, 0]]), 1),
                (to_ratio(array![[-2, 0, 0]]), 1),
                (to_ratio(array![[0, 1, 0]]), 1),
                (to_ratio(array![[1, -1, 2]]), 1),
                (to_ratio(array![[-1, 0, 2]]), 1),
                (to_ratio(array![[1, 1, -2]]), 1),
                (to_ratio(array![[-1, 2, -2]]), 1),
                (to_ratio(array![[2, -1, 0]]), 1),
                (to_ratio(array![[-2, 1, 0]]), 1),
                (to_ratio(array![[1, -2, 2]]), 1),
                (to_ratio(array![[-1, -1, 2]]), 1),
                (to_ratio(array![[1, 0, -2]]), 1),
                (to_ratio(array![[-1, 1, -2]]), 1),
                (to_ratio(array![[0, -1, 0]]), 1),
                (to_ratio(array![[1, 0, 0]]), 1),
                (to_ratio(array![[-1, 1, 0]]), 1),
                (to_ratio(array![[0, -1, 2]]), 1),
                (to_ratio(array![[0, 1, -2]]), 1),
                (to_ratio(array![[1, -1, 0]]), 1),
                (to_ratio(array![[-1, 0, 0]]), 1),
                (to_ratio(array![[0, 0, 0]]), 3),
            ];

            assert_eq!(set_diff(results.iter(), expected.iter()).len(), 0);
        }
    }

    #[test]
    fn test_xis() {
        {
            let algebra = helper_liealgebra(GroupTestType::A);
            let stabs = vec![1, 2];
            let xis = algebra.xis(&stabs);

            let expected = vec![to_ratio(array![[1, 0, 1]]), to_ratio(array![[-1, 1, 1]])];

            assert_eq!(xis, expected, "Group A xi error");
        }
        {
            let algebra = helper_liealgebra(GroupTestType::B);
            let stabs = vec![1, 2];
            let xis = algebra.xis(&stabs);

            let expected = vec![
                to_ratio(array![[0, 1, 0]]),
                to_ratio(array![[-1, 0, 2]]),
                to_ratio(array![[1, 0, 0]]),
                to_ratio(array![[-1, 1, 0]]),
            ];

            assert_eq!(xis, expected, "Group B xi error");
        }
    }

    #[test]
    fn test_xi_multiplicity() {
        {
            let algebra = helper_liealgebra(GroupTestType::A);
            let results = algebra.xi_multiplicity(to_ratio(array![[0, 1, 0]]));
            assert_eq!(
                results,
                vec![
                    (to_ratio(array![[1, 0, 1]]), 8),
                    (to_ratio(array![[0, -1, 2]]), 2),
                    (to_ratio(array![[2, -1, 0]]), 2),
                ],
                "Group A incorrecnt xi_mul"
            )
        }
        {
            let algebra = helper_liealgebra(GroupTestType::B);
            let results = algebra.xi_multiplicity(to_ratio(array![[1, 0, 0]]));

            assert_eq!(
                results,
                vec![
                    (to_ratio(array![[0, 1, 0]]), 8),
                    (to_ratio(array![[-1, 0, 2]]), 4),
                    (to_ratio(array![[1, 0, 0]]), 2),
                    (to_ratio(array![[-1, 1, 0]]), 4),
                ],
                "Group B incorrecnt xi_mul"
            )
        }
    }

    #[test]
    fn test_weight_multiplicity_highest_weight() {
        {
            let algebra = helper_liealgebra(GroupTestType::A);
            let result = algebra.weight_multiplicity_highest_weight(
                to_ratio(array![[0, 1, 0]]),
                to_ratio(array![[0, 0, 2]]),
            );
            let expected: Vec<(Array2R, Array2R, usize)> =
                vec![(to_ratio(array![[0, 0, 2]]), to_ratio(array![[0, -1, 2]]), 2)];

            assert_eq!(result, expected);
        }
        {
            let algebra = helper_liealgebra(GroupTestType::B);
            let result = algebra.weight_multiplicity_highest_weight(
                to_ratio(array![[1, 0, 0]]),
                to_ratio(array![[0, 0, 2]]),
            );
            let expected: Vec<(Array2R, Array2R, usize)> = vec![
                (to_ratio(array![[0, 0, 2]]), to_ratio(array![[-1, 0, 2]]), 4),
                (to_ratio(array![[0, 1, 0]]), to_ratio(array![[-1, 1, 0]]), 4),
            ];

            assert_eq!(result, expected);
        }
    }

    #[test]
    fn test_single_dom_weights() {
        let algebra = helper_liealgebra(GroupTestType::A);

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
                vec![to_ratio(array![[0, 1, 0]]), to_ratio(array![[2, 0, 0]])].into_iter(),
            );

            assert_eq!(result, expected)
        }
    }

    #[test]
    fn test_positive_roots() {
        let algebra = helper_liealgebra(GroupTestType::A);
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
        {
            let algebra = helper_liealgebra(GroupTestType::A);

            let result = algebra
                .weight_multiplicity(&to_ratio(array![[1, 1, 1]]), &to_ratio(array![[0, 0, 2]]));
            assert_eq!(result, 0);
        }
        {
            let algebra = helper_liealgebra(GroupTestType::B);

            let result = algebra
                .weight_multiplicity(&to_ratio(array![[1, 0, 0]]), &to_ratio(array![[0, 0, 2]]));
            assert_eq!(result, 2);
        }
    }

    #[test]
    fn test_weight_parities() {
        let algebra = helper_liealgebra(GroupTestType::A);
        let tower = algebra.weight_system_with_mul(to_ratio(array![[1, 0, 0]]));
        let results: Vec<_> = algebra.weight_parities(tower, to_ratio(array![[0, 0, 1]]));

        let expected: Vec<_> = vec![
            (1, to_ratio(array![[1, 0, 1]])),
            (1, to_ratio(array![[0, 0, 0]])),
        ];

        assert_eq!(set_diff(results.iter(), expected.iter()).len(), 0)
    }

    #[test]
    fn test_tensorproduct() {
        {
            let algebra = helper_liealgebra(GroupTestType::A);

            let decomp = algebra
                .tensor_product_decomp(to_ratio(array![[1, 0, 0]]), to_ratio(array![[1, 0, 0]]));
            let results: HashSet<_> = HashSet::from_iter(decomp.clone().into_iter());
            let expected: HashSet<_> = HashSet::from_iter(
                vec![to_ratio(array![[2, 0, 0]]), to_ratio(array![[0, 1, 0]])].into_iter(),
            );

            assert_eq!(
                results, expected,
                "Group A\nTwo term tensordecmop is not correct"
            );
        }

        {
            let algebra = helper_liealgebra(GroupTestType::B);
            let expected = algebra
                .tensor_product_decomp(to_ratio(array![[1, 0, 0]]), to_ratio(array![[1, 0, 0]]));

            let results = vec![
                to_ratio(array![[0, 0, 0]]),
                to_ratio(array![[0, 1, 0]]),
                to_ratio(array![[2, 0, 0]]),
            ];

            assert_eq!(
                set_diff(results.iter(), expected.iter()).len(),
                0,
                "Group B\nTwo term tensordecmop is not correct"
            );
        }
        {
            let algebra = helper_liealgebra(GroupTestType::B);
            let expected = algebra
                .tensor_product_decomp(to_ratio(array![[2, 0, 0]]), to_ratio(array![[1, 0, 0]]));

            let results = vec![
                to_ratio(array![[1, 0, 0]]),
                to_ratio(array![[3, 0, 0]]),
                to_ratio(array![[1, 1, 0]]),
            ];
            assert_eq!(
                set_diff(results.iter(), expected.iter()).len(),
                0,
                "Group B\nTwo term tensordecmop is not correct"
            );
        }
    }

    // #[test]
    // fn test_dim() {
    //     let algebra = helper_liealgebra(GroupTestType::A1);

    //     let irreps = algebra.dim(to_ratio(array![[2]]));
    //     let expected = 3;
    //     assert_eq!(
    //         irreps, expected,
    //         "Group A1\nTwo term test_dim is not correct"
    //     );
    // }

    // #[test]
    // fn test_get_irrep_by_dim() {
    //     {
    //         let algebra = helper_liealgebra(GroupTestType::A);

    //         let irreps = algebra.irrep_by_dim(20, 3);
    //         let expected = vec![
    //             to_ratio(array![[1, 1, 0]]),
    //             to_ratio(array![[0, 1, 1]]),
    //             to_ratio(array![[0, 2, 0]]),
    //             to_ratio(array![[3, 0, 0]]),
    //             to_ratio(array![[0, 0, 3]]),
    //         ];
    //         assert_eq!(
    //             irreps, expected,
    //             "Group A\nTwo term get_irrep_by_dim is not correct"
    //         );
    //     }
    //     {
    //         let algebra = helper_liealgebra(GroupTestType::B);

    //         let irreps = algebra.irrep_by_dim(2800, 3);
    //         let expected = vec![to_ratio(array![[1, 2, 1]])];
    //         assert_eq!(
    //             irreps, expected,
    //             "Group B\nTwo term get_irrep_by_dim is not correct"
    //         );
    //     }

    //     {
    //         let algebra = helper_liealgebra(GroupTestType::A1);

    //         let irreps = algebra.irrep_by_dim(3, 2);
    //         let expected = vec![to_ratio(array![[2]])];
    //         assert_eq!(
    //             irreps, expected,
    //             "Group A1\nTwo term get_irrep_by_dim is not correct"
    //         );
    //     }
    // }

    // #[test]
    // fn test_conjugate() {
    //     {
    //         let algebra = helper_liealgebra(GroupTestType::A);

    //         let irreps = algebra.conjugate(to_ratio(array![[1, 1, 0]]));
    //         let expected = to_ratio(array![[0, 1, 1]]);
    //         assert_eq!(
    //             irreps, expected,
    //             "Group A\nTwo term get_irrep_by_dim is not correct"
    //         );
    //     }
    // }
}
