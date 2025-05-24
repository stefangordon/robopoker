//! Subgame solving for improved strategies in critical game situations.
//!
//! This module implements unsafe subgame solving which re-solves portions of the game tree
//! with finer action abstractions to achieve better strategies in important pots.
//! 
//! ## Key Features
//! 
//! - **Enhanced Action Abstraction**: 20 bet sizes (0.2x to 10x pot) vs 2-10 in blueprint
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
//!     .with_iterations(500)
//!     .build();
//! let strategy = solver.solve().await;
//! ```

mod encoder;
mod profile;
mod solver;

pub use solver::{SubgameSolver, SubgameSolverBuilder};

// Re-export types needed by subgame solving
use crate::mccfr::nlhe::{Edge, Game, Info, Turn};
use crate::mccfr::types::policy::Policy;
use crate::gameplay::odds::Odds;

// Internal use only
use encoder::SubgameEncoder;
use profile::SubgameProfile;

/// Enhanced bet sizes for subgame solving (20 sizes from 0.2x to 10x pot)
pub const SUBGAME_RAISES: [Odds; 20] = [
    Odds(1, 5),   // 0.20x pot
    Odds(1, 4),   // 0.25x pot
    Odds(1, 3),   // 0.33x pot
    Odds(2, 5),   // 0.40x pot
    Odds(1, 2),   // 0.50x pot
    Odds(3, 5),   // 0.60x pot
    Odds(2, 3),   // 0.67x pot
    Odds(3, 4),   // 0.75x pot
    Odds(1, 1),   // 1.00x pot
    Odds(5, 4),   // 1.25x pot
    Odds(3, 2),   // 1.50x pot
    Odds(7, 4),   // 1.75x pot
    Odds(2, 1),   // 2.00x pot
    Odds(5, 2),   // 2.50x pot
    Odds(3, 1),   // 3.00x pot
    Odds(4, 1),   // 4.00x pot
    Odds(5, 1),   // 5.00x pot
    Odds(6, 1),   // 6.00x pot
    Odds(8, 1),   // 8.00x pot
    Odds(10, 1),  // 10.00x pot
];

/// Minimum iterations for subgame solving
pub const SUBGAME_MIN_ITERATIONS: usize = 100;

/// Maximum iterations for subgame solving
pub const SUBGAME_MAX_ITERATIONS: usize = 1000;

/// Convergence threshold for early stopping (sum of absolute strategy changes)
pub const SUBGAME_CONVERGENCE_THRESHOLD: f32 = 0.001; 