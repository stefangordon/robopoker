#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum Turn {
    P1,
    P2,
    Terminal,
}

impl crate::mccfr::traits::turn::Turn for Turn {
    fn chance() -> Self {
        Self::Terminal
    }
}

impl std::fmt::Display for Turn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}
