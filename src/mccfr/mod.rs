//! This module implements Counterfactual Regret Minimization (CFR) algorithms for various games.
//!
//! # Submodules
//!
//! - `nlhe`: Implementation of CFR for No-Limit Texas Hold'em poker
//! - `rps`: Simple Rock-Paper-Scissors implementation used as a toy example and test case
//! - `structs`: Core data structures used in CFR implementations
//! - `traits`: Generic traits that can be implemented for any tree-based game
//! - `types`: Type aliases and common types used across CFR implementations
//! - `subgame`: Subgame solving for improved strategies in critical situations
//! - `core`: Core module for common utilities and shared functionality
//!
//! The module provides both concrete game implementations (`nlhe`, `rps`) as well as
//! generic infrastructure (`structs`, `traits`, `types`) that can be reused for
//! implementing CFR on any extensive-form game with perfect recall.

pub mod core;
#[cfg(feature = "native")]
pub mod exploitability;
pub mod nlhe;
pub mod rps;
pub mod structs;
pub mod subgame;
pub mod traits;
pub mod types;

pub use nlhe::solver::NLHE;
pub use rps::solver::RPS;
pub use traits::blueprint::Blueprint;
