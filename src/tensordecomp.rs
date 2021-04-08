use crate::debug::debug_on;
use crate::matrix_methods::{all_pos, select_pos_diff};
use crate::orbit::OrbitMethods;
use crate::utils::{adjacent_find, Array2R};
// use itertools::Itertools;
use num::rational::Ratio;
use numpy::PyReadonlyArray3;
use std::collections::HashSet;
use std::iter::FromIterator;
use std::iter::IntoIterator;
use std::iter::Iterator;

pub struct TensorDecomposition {
    // cartan: Array2R,
    omega: Array2R,
    orbitmethods: OrbitMethods,
    positive_roots: Vec<Array2R>,
    rank: usize,
}

impl TensorDecomposition {
    pub fn new(
        simple_roots: PyReadonlyArray3<i64>,
        _: PyReadonlyArray3<i64>,
        cartan_inv: PyReadonlyArray3<i64>,
        cocartan_t: PyReadonlyArray3<i64>,
        omega: PyReadonlyArray3<i64>,
        omega_inv: PyReadonlyArray3<i64>,
        n_roots: usize,
        rank: usize,
    ) -> TensorDecomposition {
        if debug_on() {
            println!("Converting omega");
        }
        // let omega = to_rational_matrix(omega);

        if debug_on() {
            println!("Converting cartan");
        }
        // let cartan = to_rational_matrix(cartan);

        let orbitmethods = OrbitMethods::new(
            simple_roots,
            omega_inv,
            omega,
            cartan_inv,
            cocartan_t,
            n_roots,
            rank,
        );
        let positive_roots = TensorDecomposition::generate_positive_roots(&orbitmethods);

        TensorDecomposition {
            // cartan: cartan,
            omega: orbitmethods.omega.clone(),
            orbitmethods: orbitmethods,
            positive_roots: positive_roots,
            rank: rank,
        }
    }

    pub fn tensor_product_decomp(&self, weights: Vec<Array2R>) -> Vec<Array2R> {
        let mut decomp = Vec::new();
        let n = weights.len();
        for i in 0..(n - 1) {
            decomp.extend(
                self.tensor_product_decomp_double(weights[i].clone(), weights[i + 1].clone()),
            );
        }
        decomp
    }

    fn tensor_product_decomp_double(&self, irrep1: Array2R, irrep2: Array2R) -> Vec<Array2R> {
        let o1 = self.ortho_to_omega(&irrep1);
        let o2 = self.ortho_to_omega(&irrep2);

        let tower1 = self.weight_system(o1);
        let rho = Array2R::ones((1, self.rank));

        let mut weight_parities = Vec::new();

        for w in tower1.iter() {
            let (mut t, p) = self.chamber_rotate(w.clone() + o2.clone() + rho.clone());
            if t.iter().filter(|x| **x == Ratio::new(0, 1)).count() > 0 {
                t -= &rho;
                weight_parities.push((p, t.clone()));
            }
        }

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

    fn weight_system(&self, irrep: Array2R) -> Vec<Array2R> {
        let mut dom_weight_system = Vec::new();
        for x in self.single_dom_weights(&irrep).iter() {
            dom_weight_system.push((
                x.clone(),
                self.weight_multiplicity(x.clone(), irrep.clone()),
            ));
        }

        let mut weight_system = Vec::new();

        for (w, m) in dom_weight_system.iter() {
            for _ in 0..*m {
                let orbit = self.orbitmethods.orbit(w.clone());
                weight_system.extend(orbit.iter().cloned());
            }
        }
        weight_system.sort_by(|a, b| {
            let k1 = self.k_level(irrep.clone() - a);
            let k2 = self.k_level(irrep.clone() - b);

            k1.cmp(&k2)
        });

        weight_system
    }

    fn single_dom_weights<'a>(&self, irrep: &'a Array2R) -> Vec<Array2R> {
        let omega_pr: Vec<Array2R> = self
            .positive_roots
            .iter()
            .map(|x| self.ortho_to_omega(x))
            .collect();

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

    fn ortho_to_omega<'a>(&self, x: &'a Array2R) -> Array2R {
        x.dot(&self.orbitmethods.omega_inv)
    }

    fn omega_to_ortho(&self, x: Array2R) -> Array2R {
        x.dot(&self.omega)
    }

    fn omega_to_alpha<'a>(&self, x: &'a Array2R) -> Array2R {
        x.dot(&self.orbitmethods.cartan_inv)
    }

    fn k_level(&self, x: Array2R) -> Ratio<i64> {
        self.omega_to_alpha(&x).sum()
    }

    fn generate_positive_roots<'a>(orbitmethods: &'a OrbitMethods) -> Vec<Array2R> {
        let mut roots = Vec::new();

        for i in orbitmethods.simple_roots.iter() {
            let orbit = orbitmethods.orbit(i.clone());
            roots.extend(orbit.iter().cloned());
        }
        roots
    }

    fn chamber_rotate(&self, weight: Array2R) -> (Array2R, i64) {
        if all_pos(&weight) {
            return (weight, 1);
        }

        let reflection_matrices = self.orbitmethods.reflection_matrices();
        let mut reflected = vec![self.omega_to_ortho(weight)];

        let mut parity = 1;
        loop {
            let mut temp = Vec::new();
            for m in reflection_matrices.iter() {
                for r in reflected.iter() {
                    let t = r.dot(m);
                    parity *= -1;
                    let ref_omega = self.ortho_to_omega(&t);
                    if all_pos(&ref_omega) {
                        return (ref_omega, parity);
                    }
                    temp.push(t);
                }
            }
            reflected.append(&mut temp)
        }
    }

    fn weight_multiplicity(&self, weight: Array2R, irrep: Array2R) -> i64 {
        let (dom, _) = self.chamber_rotate(weight.clone());
        let dom_irrep = self.single_dom_weights(&irrep);

        if dom == irrep {
            return 1;
        }

        let k = self.k_level(irrep.clone() - weight).to_integer();

        let mut highest_weights = Vec::new();
        for i in 0..k {
            for (xi, mul) in self.xi_multiplicity(dom.clone()).iter() {
                let d = dom.clone() + xi.mapv(|x| x * (i + 1));
                if dom_irrep.contains(&d) {
                    highest_weights.push((dom.clone(), xi.clone(), mul.clone()))
                }
            }
        }

        let mut multiplicity = 0;
        let rho = Array2R::ones((1, self.rank));

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

    fn scalar_product(&self, a: Array2R, b: Array2R) -> i64 {
        self.omega_to_ortho(a).dot(&self.omega_to_ortho(b))[[0, 0]].to_integer()
    }

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

        let xis = self
            .positive_roots
            .iter()
            .map(|s| self.ortho_to_omega(s))
            .filter(all_pos);

        for xi in xis {
            let mut xi_stab = HashSet::new();
            let temp = self.omega_to_alpha(&xi);
            for (idx, m) in temp.iter().enumerate() {
                if *m > Ratio::new(0, 1) {
                    xi_stab.insert(idx);
                }
            }

            let diff = xi_stab.difference(&stabs);

            let orbit = self.orbitmethods.stable_orbit(
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
}

/// Returns the unique set of positive arrays being subtracted by `arrays`
fn union_new_weights<'a>(x: &'a HashSet<Array2R>, arrays: &'a Vec<Array2R>) -> HashSet<Array2R> {
    let mut res = HashSet::new();
    for w in x.iter() {
        res = res.union(&select_pos_diff(w, arrays)).cloned().collect();
    }
    res.iter().cloned().collect()
}
