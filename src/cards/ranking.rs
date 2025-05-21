use super::rank::Rank;

/// A poker hand's value.
///
/// This is a simplified version of the hand's value, and does not include the hand's kicker cards.
/// The value is ordered by the hand's Strength, and the kicker cards are used to break ties.
#[cfg(feature = "shortdeck")]
#[derive(Debug, Clone, Copy, Eq, PartialEq, PartialOrd, Ord)]
pub enum Ranking {
    HighCard(Rank),        // 4 kickers
    OnePair(Rank),         // 3 kickers
    TwoPair(Rank, Rank),   // 1 kickers
    ThreeOAK(Rank),        // 2 kickers
    Straight(Rank),        // 0 kickers
    Flush(Rank),           // 0 kickers
    FullHouse(Rank, Rank), // 0 kickers
    FourOAK(Rank),         // 1 kickers
    StraightFlush(Rank),   // 0 kickers
    MAX,                   // useful for showdown implementation
}
#[cfg(not(feature = "shortdeck"))]
#[derive(Debug, Clone, Copy, Eq, PartialEq, PartialOrd, Ord)]
pub enum Ranking {
    HighCard(Rank),        // 4 kickers
    OnePair(Rank),         // 3 kickers
    TwoPair(Rank, Rank),   // 1 kickers
    ThreeOAK(Rank),        // 2 kickers
    Straight(Rank),        // 0 kickers
    FullHouse(Rank, Rank), // 0 kickers
    Flush(Rank),           // 0 kickers
    FourOAK(Rank),         // 1 kickers
    StraightFlush(Rank),   // 0 kickers
    MAX,                   // useful for showdown implementation
}

impl Ranking {
    pub fn n_kickers(&self) -> usize {
        let n = match self {
            Ranking::HighCard(_) => 4,
            Ranking::OnePair(_) => 3,
            Ranking::ThreeOAK(_) => 2,
            Ranking::FourOAK(_) | Ranking::TwoPair(_, _) => 1,
            _ => 0,
        };
        n
    }
}

impl std::fmt::Display for Ranking {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Ranking::MAX => unreachable!(),
            Ranking::FullHouse(r1, r2) => write!(f, "FullHouse     {}{}", r1, r2),
            Ranking::TwoPair(r1, r2) => write!(f, "TwoPair       {}{}", r1, r2),
            Ranking::HighCard(r) => write!(f, "HighCard      {} ", r),
            Ranking::OnePair(r) => write!(f, "OnePair       {} ", r),
            Ranking::ThreeOAK(r) => write!(f, "ThreeOfAKind  {} ", r),
            Ranking::Straight(r) => write!(f, "Straight      {} ", r),
            Ranking::FourOAK(r) => write!(f, "FourOfAKind   {} ", r),
            Ranking::Flush(r) => write!(f, "Flush         {} ", r),
            Ranking::StraightFlush(r) => write!(f, "StraightFlush {} ", r),
        }
    }
}
