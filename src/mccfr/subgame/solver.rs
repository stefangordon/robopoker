use super::{Edge, Game, Info, Turn};
use super::{SUBGAME_MIN_ITERATIONS, SUBGAME_MAX_ITERATIONS};
use crate::analysis::response::Decision;
use crate::cards::isomorphism::Isomorphism;
use crate::clustering::abstraction::Abstraction;
use crate::mccfr::structs::tree::Tree;
use crate::mccfr::traits::encoder::Encoder;
use crate::mccfr::traits::profile::Profile;
use crate::mccfr::types::counterfactual::Counterfactual;
use rayon::prelude::*;
use std::collections::BTreeMap;
use crate::mccfr::nlhe::encoder::Encoder as GameEncoder;
use crate::mccfr::nlhe::profile::{Profile as NLProfile, ProfileBuilder};
use super::SubgameSizer;
use crate::mccfr::traits::info::Info as InfoTrait;
type SGEncoder = GameEncoder<SubgameSizer>;

/// Subgame solver for re-solving game trees with enhanced action abstraction
pub struct SubgameSolver {
    profile: NLProfile,
    warm_start: Vec<Decision>,
    encoder: SGEncoder,
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
        let (profile, warm) = ProfileBuilder::new()
            .with_warm_start(warm_start)
            .with_regret_bounds(-1.0e7, 1.0e7)
            .build();

        Self {
            profile,
            warm_start: warm,
            encoder: SGEncoder::new(abstraction_lookup),
            root_game,
            iterations: iterations.clamp(SUBGAME_MIN_ITERATIONS, SUBGAME_MAX_ITERATIONS),
        }
    }

    /// Solve the subgame and return the strategy
    pub async fn solve(mut self) -> Vec<Decision> {
        let start_time = std::time::Instant::now();

        // Set the root info and seed warm-start if provided
        let root_info = self.encoder.seed(&self.root_game);
        if !self.warm_start.is_empty() {
            self.profile.seed_decisions(&root_info, &self.warm_start);
        }

        let mut last_strategy = self.root_strategy(&root_info);
        let mut converged_at = None;

        log::debug!("Initial subgame strategy: {} actions", last_strategy.len());
        if log::log_enabled!(log::Level::Trace) {
            for (i, decision) in last_strategy.iter().enumerate() {
                log::trace!("  Action {}: edge={:?} weight={:.4}", i, decision.edge(), decision.weight());
            }
        }

        // Process iterations in chunks for better parallelization
        let chunk_size = 5;
        let mut i = 0;

        while i < self.iterations {
            let remaining = self.iterations - i;
            let current_chunk_size = chunk_size.min(remaining);
            let mut stable = 0u8;

            // Process multiple iterations in parallel
            let all_updates: Vec<_> = (0..current_chunk_size)
                .into_par_iter()
                .flat_map(|_| self.batch())
                .collect();

            // Apply all updates at once
            self.profile.apply_updates(all_updates);
            for _ in 0..current_chunk_size {
                self.profile.increment();
            }

            i += current_chunk_size;

            // Check for convergence every 10 iterations, but only after minimum iterations
            if i >= 50 && i % 10 == 0 {  // Don't check convergence until at least 50 iterations
                let current_strategy = self.root_strategy(&root_info);
                if Self::has_converged(&last_strategy, &current_strategy, &mut stable) {
                    converged_at = Some(i);
                    break;
                }
                last_strategy = current_strategy;
            }
        }

        let elapsed = start_time.elapsed();
        match converged_at {
            Some(iterations) => {
                log::info!("Subgame converged: {} iterations in {:.1}ms", iterations, elapsed.as_secs_f64() * 1000.0);
            }
            None => {
                log::info!("Subgame completed: {} iterations in {:.1}ms (no convergence)", self.iterations, elapsed.as_secs_f64() * 1000.0);
            }
        }

        self.root_strategy(&root_info)
    }

    /// Generate a batch of counterfactual updates
    fn batch(&self) -> Vec<Counterfactual<Edge, Info>> {
        let walker = self.profile.walker();
        // Adaptive batch size based on CPU cores for optimal parallelization
        let batch_size = (rayon::current_num_threads() * 16).max(64).min(256);

        let updates = (0..batch_size)
            .into_par_iter()
            .map(|_| self.build_tree())
            .flat_map_iter(move |tree| {
                tree.partition()
                    .into_values()
                    .into_iter()
                    .filter(move |infoset| infoset.head().game().turn() == walker)
            })
            .map(|infoset| self.counterfactual(&infoset))
            .collect::<Vec<_>>();

        log::trace!("Generated {} counterfactual updates from {} trees", updates.len(), batch_size);
        updates
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
        let mut nodes_built = 1; // Count the root
        while let Some(leaf) = todo.pop() {
            let info = self.encoder.info(&tree, leaf);
            let node = tree.grow(info, leaf);
            nodes_built += 1;
            let children = self.encoder.branches(&node);
            let children = self.profile.explore(&node, children);
            todo.extend(children);
        }

        log::trace!("Built tree with {} nodes", nodes_built);
        tree
    }

    /// Compute counterfactual values for an infoset
    fn counterfactual(&self, infoset: &crate::mccfr::structs::infoset::InfoSet<Turn, Edge, Game, Info>) -> Counterfactual<Edge, Info> {
        let regret_vec = self.profile.regret_vector(infoset);
        let policy_vec = self.profile.policy_vector(infoset);

        if log::log_enabled!(log::Level::Trace) {
            let regret_sum: f32 = regret_vec.iter().map(|(_, r)| r.abs()).sum();
            let policy_sum: f32 = policy_vec.iter().map(|(_, p)| *p).sum();
            log::trace!("Counterfactual: {} regrets (sum={:.6}), {} policies (sum={:.6})",
                       regret_vec.len(), regret_sum, policy_vec.len(), policy_sum);
        }

        (infoset.info(), regret_vec, policy_vec)
    }

    /// Return true when the root behaviour strategy has stabilised.
    /// - `stable` is a small counter kept by the caller.
    fn has_converged(old: &[Decision], new: &[Decision], stable: &mut u8) -> bool {
        if old.len() != new.len() { *stable = 0; return false; }

        // Largest single-action change (L∞)
        let mut max_diff = 0f32;
        for (o, n) in old.iter().zip(new) {
            max_diff = max_diff.max((o.weight() - n.weight()).abs());
        }

        const MAX_EPS: f32 = 1e-3;   // 0.1 %

        if max_diff < MAX_EPS {
            *stable += 1;            // one more stable check
        } else {
            *stable = 0;             // reset streak
        }

        log::debug!("Δ_max {:.5}  stable {}", max_diff, *stable);
        *stable >= 2                // stop after 2 consecutive passes
    }

    /// Extract root strategy using positive-regret matching with fallbacks.
    fn root_strategy(&self, info: &Info) -> Vec<Decision> {
        let edges: Vec<(Edge, f32)> = info
            .choices()
            .into_iter()
            .map(|edge| {
                let regret = self.profile.sum_regret(info, &edge);
                (edge, regret)
            })
            .collect();

        let pos_sum: f32 = edges.iter().map(|(_, r)| if *r > 0.0 { *r } else { 0.0 }).sum();

        if pos_sum > 0.0 {
            edges
                .into_iter()
                .map(|(edge, r)| Decision::from((edge, if r > 0.0 { r / pos_sum } else { 0.0 })))
                .collect()
        } else {
            // Fall back to accumulated policy σ̄
            let weights: Vec<(Edge, f32)> = info
                .choices()
                .into_iter()
                .map(|edge| {
                    let w = self.profile.sum_policy(info, &edge);
                    (edge, w)
                })
                .collect();
            let sum_policy: f32 = weights.iter().map(|(_, w)| *w).sum();
            if sum_policy > 0.0 {
                weights
                    .into_iter()
                    .map(|(edge, w)| Decision::from((edge, w / sum_policy)))
                    .collect()
            } else if !self.warm_start.is_empty() {
                self.warm_start.clone()
            } else {
                let n = info.choices().len() as f32;
                info.choices()
                    .into_iter()
                    .map(|edge| Decision::from((edge, 1.0 / n)))
                    .collect()
            }
        }
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