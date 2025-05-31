//! Subgame solving for improved strategies in critical game situations.
//!
//! This module implements unsafe subgame solving which re-solves portions of the game tree
//! with finer action abstractions to achieve better strategies in important pots.
//!
//! ## Key Features
//!
//! - **Enhanced Action Abstraction**: 10 bet sizes (0.25x to 4x pot) vs 2-5 in blueprint
//! - **Warm-Starting**: Initializes from blueprint strategy for faster convergence
//! - **Early Stopping**: Detects convergence based on strategy change threshold
//! - **Memory Efficient**: Only requires blueprint query for current position
//!
//! ## Usage
//!
//! Subgame solving is automatically triggered by the API for important spots:
//!
//! ```rust,ignore
//! // Automatic triggers:
//! // - River: pot > 20 chips
//! // - Turn: pot > 40 chips and SPR < 2.0
//! let strategy = api.policy(recall).await?;
//!
//! // Or use directly:
//! let solver = SubgameSolver::builder()
//!     .with_game_state(game)
//!     .with_warm_start(blueprint_strategy)
//!     .with_iterations(1000)
//!     .build();
//! let strategy = solver.solve().await;
//! ```

#[cfg(feature = "subgame")]
mod solver;

#[cfg(feature = "subgame")]
pub use solver::{SubgameSolver, SubgameSolverBuilder};

// Re-export types needed by subgame solving
use crate::mccfr::nlhe::{Game};
use crate::gameplay::odds::Odds;

// Internal use only
use crate::mccfr::nlhe::encoder::BetSizer;

/// Enhanced bet sizes for subgame solving using the full preflop grid (10 sizes from 0.25x to 4x pot)
/// This provides 2-5x more granularity than the typical post-flop grids used in the blueprint
#[cfg(not(feature = "shortdeck"))]
pub const SUBGAME_RAISES: [Odds; 10] = Odds::GRID;

#[cfg(feature = "shortdeck")]
pub const SUBGAME_RAISES: [Odds; 5] = Odds::GRID;

/// Minimum iterations for subgame solving
pub const SUBGAME_MIN_ITERATIONS: usize = 100;

/// Maximum iterations for subgame solving
pub const SUBGAME_MAX_ITERATIONS: usize = 1000;

/// Convergence threshold for early stopping (sum of absolute strategy changes)
pub const SUBGAME_CONVERGENCE_THRESHOLD: f32 = 0.001;

/// Numeric parameters specific to subgame solving (so we don't disturb global training constants)
#[derive(Clone, Copy, Debug)]
pub struct SubgameParams {
    pub regret_min: crate::Utility,
    pub regret_max: crate::Utility,
    pub sampling_eps: crate::Probability,
}

impl Default for SubgameParams {
    fn default() -> Self {
        Self {
            regret_min: -1.0e7,  // looser bounds than global -3e5
            regret_max:  1.0e7,
            sampling_eps: 1e-6,  // floor to stabilise importance weights
        }
    }
}

/// Marker type for subgame bet-sizing strategy
pub struct SubgameSizer;

impl BetSizer for SubgameSizer {
    fn raises(_game: &Game, depth: usize) -> Vec<Odds> {
        if depth > crate::MAX_RAISE_REPEATS {
            vec![]
        } else {
            SUBGAME_RAISES.to_vec()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gameplay::edge::Edge as GameplayEdge;

    #[test]
    fn test_subgame_raises_fit_in_edge_encoding() {
        // Verify all subgame raises can be converted to/from u8
        for odds in SUBGAME_RAISES.iter() {
            let edge = GameplayEdge::Raise(*odds);
            let encoded = u8::from(edge);
            let decoded = GameplayEdge::from(encoded);
            assert_eq!(edge, decoded, "Failed to round-trip {:?}", odds);
        }
    }
}