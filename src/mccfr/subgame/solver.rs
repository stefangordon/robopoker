use super::{Edge, Game, Info, Turn, SubgameEncoder, SubgameProfile};
use super::{SUBGAME_MIN_ITERATIONS, SUBGAME_MAX_ITERATIONS, SUBGAME_CONVERGENCE_THRESHOLD};
use crate::analysis::response::Decision;
use crate::cards::isomorphism::Isomorphism;
use crate::clustering::abstraction::Abstraction;
use crate::mccfr::structs::tree::Tree;
use crate::mccfr::traits::encoder::Encoder;
use crate::mccfr::traits::profile::Profile;
use crate::mccfr::types::counterfactual::Counterfactual;
use rayon::prelude::*;
use std::collections::BTreeMap;

/// Subgame solver for re-solving game trees with enhanced action abstraction
pub struct SubgameSolver {
    profile: SubgameProfile,
    encoder: SubgameEncoder,
    root_game: Game,
    iterations: usize,
}

impl SubgameSolver {
    pub fn new(
        root_game: Game,
        warm_start: Vec<Decision>,
        iterations: usize,
        abstraction_lookup: BTreeMap<Isomorphism, Abstraction>,
    ) -> Self {
        Self {
            profile: SubgameProfile::new(warm_start),
            encoder: SubgameEncoder::new(abstraction_lookup),
            root_game,
            iterations: iterations.clamp(SUBGAME_MIN_ITERATIONS, SUBGAME_MAX_ITERATIONS),
        }
    }

    /// Solve the subgame and return the strategy
    pub async fn solve(mut self) -> Vec<Decision> {
        log::info!("Starting subgame solving for {} iterations", self.iterations);
        
        let mut last_strategy = self.profile.get_strategy_for_root();
        
        for i in 0..self.iterations {
            // Generate batch of trees and compute updates
            let updates = self.batch();
            
            // Apply updates to profile
            self.profile.apply_updates(updates);
            self.profile.increment();
            
            // Check for convergence every 10 iterations
            if i > 0 && i % 10 == 0 {
                let current_strategy = self.profile.get_strategy_for_root();
                if Self::has_converged(&last_strategy, &current_strategy) {
                    log::info!("Subgame converged after {} iterations", i);
                    break;
                }
                last_strategy = current_strategy;
            }
        }
        
        self.profile.get_strategy_for_root()
    }

    /// Generate a batch of counterfactual updates
    fn batch(&self) -> Vec<Counterfactual<Edge, Info>> {
        let walker = self.profile.walker();
        let batch_size = 32; // Smaller batch size for subgames
        
        (0..batch_size)
            .into_par_iter()
            .map(|_| self.build_tree())
            .flat_map_iter(move |tree| {
                tree.partition()
                    .into_values()
                    .into_iter()
                    .filter(move |infoset| infoset.head().game().turn() == walker)
            })
            .map(|infoset| self.counterfactual(&infoset))
            .collect()
    }

    /// Build a game tree from the subgame root
    fn build_tree(&self) -> Tree<Turn, Edge, Game, Info> {
        let mut tree = Tree::default();
        let mut todo = Vec::new();
        
        // Start from subgame root
        let info = self.encoder.seed(&self.root_game);
        let node = tree.seed(info, self.root_game.clone());
        
        let children = self.encoder.branches(&node);
        let children = self.profile.explore(&node, children);
        todo.extend(children);
        
        // Build tree with subgame action abstraction
        while let Some(leaf) = todo.pop() {
            let info = self.encoder.info(&tree, leaf);
            let node = tree.grow(info, leaf);
            let children = self.encoder.branches(&node);
            let children = self.profile.explore(&node, children);
            todo.extend(children);
        }
        
        tree
    }

    /// Compute counterfactual values for an infoset
    fn counterfactual(&self, infoset: &crate::mccfr::structs::infoset::InfoSet<Turn, Edge, Game, Info>) -> Counterfactual<Edge, Info> {
        (
            infoset.info(),
            self.profile.regret_vector(infoset),
            self.profile.policy_vector(infoset),
        )
    }

    /// Check if the strategy has converged
    fn has_converged(old_strategy: &[Decision], new_strategy: &[Decision]) -> bool {
        if old_strategy.len() != new_strategy.len() {
            return false;
        }
        
        let total_change: f32 = old_strategy.iter()
            .zip(new_strategy.iter())
            .map(|(old, new)| (old.weight() - new.weight()).abs())
            .sum();
            
        total_change < SUBGAME_CONVERGENCE_THRESHOLD
    }

    /// Builder pattern methods for convenience
    pub fn builder() -> SubgameSolverBuilder {
        SubgameSolverBuilder::default()
    }
}

/// Builder for SubgameSolver with fluent API
#[derive(Default)]
pub struct SubgameSolverBuilder {
    root_game: Option<Game>,
    warm_start: Vec<Decision>,
    iterations: usize,
    abstraction_lookup: BTreeMap<Isomorphism, Abstraction>,
}

impl SubgameSolverBuilder {
    pub fn with_game_state(mut self, game: Game) -> Self {
        self.root_game = Some(game);
        self
    }

    pub fn with_warm_start(mut self, strategy: Vec<Decision>) -> Self {
        self.warm_start = strategy;
        self
    }

    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    pub fn with_abstraction_lookup(mut self, lookup: BTreeMap<Isomorphism, Abstraction>) -> Self {
        self.abstraction_lookup = lookup;
        self
    }

    pub fn build(self) -> SubgameSolver {
        SubgameSolver::new(
            self.root_game.unwrap_or_else(Game::root),
            self.warm_start,
            self.iterations.max(SUBGAME_MIN_ITERATIONS),
            self.abstraction_lookup,
        )
    }
} 