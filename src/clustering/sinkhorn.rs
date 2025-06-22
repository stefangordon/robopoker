use super::abstraction::Abstraction;
use super::histogram::Histogram;
use super::metric::Metric;
use super::potential::Potential;
use crate::transport::coupling::Coupling;
use crate::transport::density::Density;
use crate::transport::measure::Measure;
use crate::Energy;
use crate::Entropy;
use std::collections::BTreeMap;

/// using this to represent an arbitrary instance of the Kontorovich-Rubinstein
/// potential formulation of the optimal transport problem.
pub struct Sinkhorn<'a> {
    metric: &'a Metric,
    mu: &'a Histogram,
    nu: &'a Histogram,
    lhs: Potential,
    rhs: Potential,
}

impl Sinkhorn<'_> {
    /// calculate ε-minimizing coupling by scaling potentials
    fn sinkhorn(&mut self) {
        #[allow(unused)]
        for t in 0..self.iterations() {
            let ref mut next = self.lhs();
            let ref mut prev = self.lhs;
            let lhs_err = Self::delta(prev, next);
            std::mem::swap(prev, next);
            let ref mut next = self.rhs();
            let ref mut prev = self.rhs;
            let rhs_err = Self::delta(prev, next);
            std::mem::swap(prev, next);
            if lhs_err + rhs_err < self.tolerance() {
                break;
            }
        }
    }
    /// calculate next iteration of LHS and RHS potentials after Sinkhorn scaling
    fn lhs(&self) -> Potential {
        let mut map = BTreeMap::new();
        for &x in self.lhs.support() {
            let val = self.divergence(&x, &self.mu, &self.rhs);
            assert!(val.is_finite(), "lhs entropy overflow");
            map.insert(x, val);
        }
        Potential::from(map)
    }
    /// calculate next iteration of LHS and RHS potentials after Sinkhorn scaling
    fn rhs(&self) -> Potential {
        let mut map = BTreeMap::new();
        for &x in self.rhs.support() {
            let val = self.divergence(&x, &self.nu, &self.lhs);
            assert!(val.is_finite(), "rhs entropy overflow");
            map.insert(x, val);
        }
        Potential::from(map)
    }
    /// the coupling formed by joint distribution of LHS and RHS potentials
    fn coupling(&self, x: &Abstraction, y: &Abstraction) -> Energy {
        (self.lhs.density(x) + self.rhs.density(y) - self.regularization(x, y)).exp()
    }
    /// update the potential energy on a given side
    /// histogram is where a: Abstraction is supported
    /// potential is the distribution that is being integrated against
    /// so we scale PDF(A::histogram | t) by the mass of the PDF(B::potential | t, x == a)
    /// not sure yet why i'm calling it entropy but it's giving partition function.
    /// actually now that i think of it this might be KL div / relative entropy
    fn divergence(&self, x: &Abstraction, histogram: &Histogram, potential: &Potential) -> Entropy {
        let mut z = 0.0;
        for y in potential.support() {
            let e = (potential.density(y) - self.regularization(x, y)).exp();
            z += e.max(Energy::MIN_POSITIVE);
        }
        histogram.density(x).ln() - z.ln()
    }
    /// distance in fixed temperature exponent space
    fn regularization(&self, x: &Abstraction, y: &Abstraction) -> Entropy {
        self.metric.distance(x, y) / self.temperature()
    }
    /// stopping criteria
    fn delta(prev: &Potential, next: &Potential) -> Energy {
        let mut sum = 0.0;
        for x in prev.support() {
            let diff = next.density(x).exp() - prev.density(x).exp();
            sum += diff.abs();
        }
        sum
    }
    /// hyperparameter that determines strength of entropic regularization. incorrect units but whatever
    const fn temperature(&self) -> Entropy {
        crate::SINKHORN_TEMPERATURE
    }
    /// hyperparameter that determines maximum number of iterations
    const fn iterations(&self) -> usize {
        crate::SINKHORN_ITERATIONS
    }
    /// hyperparameter that determines stopping criteria
    const fn tolerance(&self) -> Energy {
        crate::SINKHORN_TOLERANCE
    }
}

impl Coupling for Sinkhorn<'_> {
    type X = Abstraction;
    type Y = Abstraction;
    type P = Potential;
    type Q = Potential;
    type M = Metric;

    fn minimize(mut self) -> Self {
        self.sinkhorn();
        self
    }
    fn flow(&self, x: &Self::X, y: &Self::Y) -> Energy {
        self.coupling(x, y) * self.metric.distance(x, y)
    }
    fn cost(&self) -> Energy {
        self.lhs
            .support()
            .flat_map(|x| self.rhs.support().map(move |y| (x, y)))
            .map(|(x, y)| self.flow(x, y))
            .inspect(|x| assert!(x.is_finite()))
            .sum::<Energy>()
    }
}

impl<'a> From<(&'a Histogram, &'a Histogram, &'a Metric)> for Sinkhorn<'a> {
    fn from((mu, nu, metric): (&'a Histogram, &'a Histogram, &'a Metric)) -> Self {
        Self {
            metric,
            mu,
            nu,
            lhs: Potential::uniform(mu),
            rhs: Potential::uniform(nu),
        }
    }
}
