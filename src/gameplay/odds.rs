use crate::Arbitrary;
use crate::Chips;
use crate::Probability;
use crate::Utility;

/// pot-normalized odds for a given raise size
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq, Ord, PartialOrd)]
pub struct Odds(pub Chips, pub Chips);

impl From<Odds> for Probability {
    fn from(odds: Odds) -> Self {
        odds.0 as Probability / odds.1 as Probability
    }
}

impl From<(Chips, Chips)> for Odds {
    fn from((a, b): (Chips, Chips)) -> Self {
        let (a, b) = Self::gcd(a, b);
        Self(a, b)
    }
}

impl Odds {
    fn gcd(a: Chips, b: Chips) -> (Chips, Chips) {
        let (mut a, mut b) = (a, b);
        while b != 0 {
            (a, b) = (b, a % b);
        }
        (a, b)
    }
    /// Pick the odds from `candidates` whose pot-ratio is closest to `amount / pot`.
    /// Assumes `candidates` is non-empty and pre-sorted by ratio.
    pub fn nearest_in(candidates: &[Self], amount: Chips, pot: Chips) -> Self {
        use crate::Probability;
        let target = amount as Utility / pot as Utility;
        candidates
            .iter()
            .min_by(|x, y| {
                let dx = (Probability::from(**x) as Utility - target).abs();
                let dy = (Probability::from(**y) as Utility - target).abs();
                dx.partial_cmp(&dy).unwrap()
            })
            .copied()
            .unwrap_or(Self::from((amount, pot)))
    }
    pub fn nearest((a, b): (Chips, Chips)) -> Self {
        let odds = a as Utility / b as Utility;
        Odds::GRID[Odds::GRID
            .map(|o| Probability::from(o)) // pre-sorted
            .binary_search_by(|p| p.partial_cmp(&odds).expect("not NaN"))
            .unwrap_or_else(|i| i.saturating_sub(1))]
    }

    #[cfg(not(feature = "shortdeck"))]
    pub const GRID: [Self; 10] = Self::PREF_RAISES;

    #[cfg(feature = "shortdeck")]
    pub const GRID: [Self; 3] = Self::PREF_RAISES;

    #[cfg(not(feature = "shortdeck"))]
    pub const PREF_RAISES: [Self; 10] = [
        Self(1, 4), // 0.25
        Self(1, 3), // 0.33
        Self(1, 2), // 0.50
        Self(2, 3), // 0.66
        Self(3, 4), // 0.75
        Self(1, 1), // 1.00
        Self(3, 2), // 1.50
        Self(2, 1), // 2.00
        Self(3, 1), // 3.00
        Self(4, 1), // 4.00
    ];

    #[cfg(feature = "shortdeck")]
    pub const PREF_RAISES: [Self; 3] = [
        Self(1, 2), // 0.50
        Self(1, 1), // 1.00
        Self(2, 1), // 2.00
    ];

    #[cfg(not(feature = "shortdeck"))]
    pub const FLOP_RAISES: [Self; 5] = [
        Self(1, 2), // 0.50
        Self(3, 4), // 0.75
        Self(1, 1), // 1.00
        Self(3, 2), // 1.50
        Self(2, 1), // 2.00
    ];

    #[cfg(feature = "shortdeck")]
    pub const FLOP_RAISES: [Self; 3] = [
        Self(1, 2), // 0.50
        Self(1, 1), // 1.00
        Self(2, 1), // 2.00
    ];

    #[cfg(not(feature = "shortdeck"))]
    pub const LATE_RAISES: [Self; 4] = [
        Self(1, 2), // 0.50
        Self(1, 1), // 1.00
        Self(3, 2), // 1.50
        Self(2, 1), // 2.00
    ];

    #[cfg(feature = "shortdeck")]
    pub const LATE_RAISES: [Self; 2] = [
        Self(1, 1), // 1.00
        Self(2, 1), // 2.00
    ];

    #[cfg(not(feature = "shortdeck"))]
    pub const LAST_RAISES: [Self; 2] = [
        Self(1, 1), // 1.00
        Self(3, 2), // 1.50
    ];

    #[cfg(feature = "shortdeck")]
    pub const LAST_RAISES: [Self; 1] = [
        Self(1, 1), // 1.00
    ];
}

impl std::fmt::Display for Odds {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let p = Probability::from(*self);
        if p > 1.0 {
            write!(f, "-{}", (p * 1.0).round() as i32)
        } else {
            write!(f, "+{}", (1.0 / p).round() as i32)
        }
    }
}

impl Arbitrary for Odds {
    fn random() -> Self {
        use rand::seq::SliceRandom;
        Self::GRID
            .choose(&mut rand::thread_rng())
            .copied()
            .expect("GRID is empty")
    }
}
