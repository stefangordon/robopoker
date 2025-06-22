use super::abstraction::Abstraction;
use super::histogram::Histogram;
use crate::transport::density::Density;
use crate::Entropy;
use crate::Probability;
use std::collections::BTreeMap;
use std::ops::AddAssign;

/// using this to represent an arbitrary instance of the Kontorovich-Rubinstein
/// potential formulation of the optimal transport problem.
/// this structure can also be treated as a normalized distribution over Abstractions.
pub struct Potential {
    map: BTreeMap<Abstraction, Entropy>,
    cache: Vec<Entropy>, // indexed by Abstraction::index(); f32::NEG_INFINITY for missing
}

impl Potential {
    /// useful for Heuristic where we don't need to allocate.
    /// i guess we don't need to allocate in Sinkhorn either. but it's
    /// nbd, + we might want to calaculate deltas between new and old potentials
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&Abstraction, &mut Entropy)> {
        self.map.iter_mut()
    }

    /// also only useful for Heuristic
    pub fn values(&self) -> impl Iterator<Item = &Entropy> {
        self.map.values()
    }

    pub fn increment(&mut self, i: &Abstraction, delta: Entropy) {
        if let Some(e) = self.map.get_mut(i) {
            e.add_assign(delta);
            self.cache[i.index()] = *e;
        } else {
            panic!("fixed abstraction space");
        }
    }

    /// zero potential over the support, in log prob space
    pub fn zeroes(h: &Histogram) -> Self {
        Self::from(
            h.support()
                .copied()
                .map(|x| (x, 0.))
                .collect::<BTreeMap<_, _>>()
        )
    }

    /// uniform distribution over the support, in log prob space
    pub fn uniform(h: &Histogram) -> Self {
        Self::from(
            h.support()
                .copied()
                .map(|x| (x, h.n()))
                .map(|(x, y)| (x, 1. / y as Probability))
                .map(|(x, y)| (x, y.ln() as Entropy))
                .collect::<BTreeMap<_, _>>()
        )
    }

    /// unit normalized distribution over the support
    pub fn normalize(h: &Histogram) -> Self {
        Self::from(
            h.support()
                .copied()
                .map(|x| (x, h.density(&x)))
                .collect::<BTreeMap<_, _>>()
        )
    }
}

impl From<BTreeMap<Abstraction, Entropy>> for Potential {
    fn from(potential: BTreeMap<Abstraction, Entropy>) -> Self {
        assert!(!potential.is_empty());
        let max_index = potential
            .keys()
            .map(|a| a.index())
            .max()
            .unwrap_or(0);
        let mut cache = vec![f32::NEG_INFINITY; max_index + 1];
        for (abs, ent) in potential.iter() {
            cache[abs.index()] = *ent;
        }
        Self { map: potential, cache }
    }
}

impl Density for Potential {
    type Support = Abstraction;
    fn density(&self, x: &Self::Support) -> Entropy {
        // Fast path using dense cache when index is within bounds.
        let idx = x.index();
        if idx < self.cache.len() {
            let p = self.cache[idx];
            assert!(p.is_finite(), "density overflow");
            return p;
        }
        // Fallback to map (should be rare).
        *self
            .map
            .get(x)
            .inspect(|p| assert!(p.is_finite(), "density overflow"))
            .expect("abstraction in potential")
    }
    fn support(&self) -> impl Iterator<Item = &Self::Support> {
        self.map.keys()
    }
}
