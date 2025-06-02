use crate::gameplay::edge::Edge;
use crate::mccfr::nlhe::Game;
use crate::mccfr::nlhe::info::Info;
use crate::gameplay::turn::Turn;
use super::{SUBGAME_MIN_ITERATIONS, SUBGAME_MAX_ITERATIONS};
use crate::mccfr::types::decision::Decision;
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
            // Translate blueprint strategy to subgame action space
            let subgame_edges = root_info.choices();
            let translated_strategy = self.translate_warm_start_to_subgame(&subgame_edges);

            if !translated_strategy.is_empty() {
                log::debug!("Translated warm start: {} blueprint actions -> {} subgame actions",
                           self.warm_start.len(), translated_strategy.len());
                self.profile.seed_decisions(&root_info, &translated_strategy);
            } else {
                log::warn!("Failed to translate warm start strategy to subgame action space");
            }
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
        let mut stable = 0u8;

        while i < self.iterations {
            let remaining = self.iterations - i;
            let current_chunk_size = chunk_size.min(remaining);

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

    /// Translate warm start strategy from blueprint action space to subgame action space
    fn translate_warm_start_to_subgame(&self, subgame_edges: &[Edge]) -> Vec<Decision> {
        let mut translated = Vec::new();

        // Log the original warm start strategy
        if log::log_enabled!(log::Level::Debug) {
            log::debug!("Original warm start strategy ({} actions):", self.warm_start.len());
            for (i, decision) in self.warm_start.iter().enumerate() {
                log::debug!("  {}: edge={:?} weight={:.6}", i, decision.edge(), decision.weight());
            }
        }

        // Group warm start decisions by action type
        let mut check_weight = 0.0;
        let mut fold_weight = 0.0;
        let mut call_weight = 0.0;
        let mut raise_weight = 0.0;
        let mut shove_weight = 0.0;
        let mut draw_weight = 0.0;

        for decision in &self.warm_start {
            match decision.edge() {
                Edge::Check => check_weight += decision.weight(),
                Edge::Fold => fold_weight += decision.weight(),
                Edge::Call => call_weight += decision.weight(),
                Edge::Raise(_) => raise_weight += decision.weight(), // Aggregate all raises
                Edge::Shove => shove_weight += decision.weight(),
                Edge::Draw => draw_weight += decision.weight(),
            }
        }

        // Log aggregated weights by action type
        log::debug!("Aggregated blueprint weights: Check={:.6} Fold={:.6} Call={:.6} Raise={:.6} Shove={:.6} Draw={:.6}",
                   check_weight, fold_weight, call_weight, raise_weight, shove_weight, draw_weight);

        // Count total raise edges in subgame
        let raise_edge_count = subgame_edges.iter()
            .filter(|e| matches!(e, Edge::Raise(_)))
            .count() as f32;

        log::debug!("Subgame has {} raise edges out of {} total edges", raise_edge_count, subgame_edges.len());

        // Map to subgame edges
        for &edge in subgame_edges {
            let weight = match edge {
                Edge::Check => check_weight,
                Edge::Fold => fold_weight,
                Edge::Call => call_weight,
                Edge::Shove => shove_weight,
                Edge::Draw => draw_weight,
                Edge::Raise(odds) => {
                    if raise_weight > 0.0 && raise_edge_count > 0.0 {
                        // Distribute raise weight across all subgame raise sizes
                        // Give more weight to middle-sized bets (around 0.75x to 1.5x pot)
                        let pot_ratio = odds.0 as f32 / odds.1 as f32;
                        let preference = if pot_ratio >= 0.5 && pot_ratio <= 2.0 {
                            1.5 // Prefer reasonable bet sizes
                        } else if pot_ratio >= 0.25 && pot_ratio <= 4.0 {
                            1.0 // Normal weight for extreme sizes
                        } else {
                            0.5 // Lower weight for very extreme sizes
                        };

                        let distributed_weight = (raise_weight * preference) / raise_edge_count;
                        log::debug!("  Raise {:?} (ratio={:.2}): preference={:.1} -> weight={:.6}",
                                   odds, pot_ratio, preference, distributed_weight);
                        distributed_weight
                    } else {
                        // Blueprint has no raises, give subgame raises small weight
                        let total_non_raise_weight = check_weight + fold_weight + call_weight + shove_weight + draw_weight;
                        if total_non_raise_weight > 0.0 && raise_edge_count > 0.0 {
                            let fallback_weight = (total_non_raise_weight * 0.1) / raise_edge_count;
                            log::debug!("  Raise {:?}: fallback weight={:.6}", odds, fallback_weight);
                            fallback_weight
                        } else {
                            0.0
                        }
                    }
                }
            };

            if weight > 0.0 {
                translated.push(Decision::from((edge, weight)));
            }
        }

        // Normalize weights to sum to 1.0
        let total_weight: f32 = translated.iter().map(|d| d.weight()).sum();
        log::debug!("Pre-normalization total weight: {:.6}", total_weight);

        if total_weight > 0.0 {
            let normalized = translated.into_iter().map(|d| {
                Decision::from((d.edge(), d.weight() / total_weight))
            }).collect::<Vec<_>>();

            // Log final translated strategy
            if log::log_enabled!(log::Level::Debug) {
                log::debug!("Final translated strategy ({} actions):", normalized.len());
                for (i, decision) in normalized.iter().enumerate() {
                    log::debug!("  {}: edge={:?} weight={:.6}", i, decision.edge(), decision.weight());
                }
            }

            normalized
        } else {
            log::warn!("Translation resulted in zero total weight!");
            Vec::new()
        }
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

        const MAX_EPS: f32 = 0.005;   // 0.1 %

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
        // Get all possible choices for this info
        let all_choices = info.choices();
        if all_choices.is_empty() {
            return Vec::new();
        }
        
        // Get all regrets in a single batch operation
        let all_regrets = self.profile.get_all_regrets(info);
        
        // Calculate sum of positive regrets across ALL possible choices
        let mut pos_sum = 0.0f32;
        let mut regrets_with_edges = Vec::with_capacity(all_choices.len());
        
        for choice_edge in all_choices.iter() {
            let regret = all_regrets
                .iter()
                .find_map(|(e, r)| if e == choice_edge { Some(*r) } else { None })
                .unwrap_or(0.0);
            
            regrets_with_edges.push((*choice_edge, regret));
            if regret > 0.0 {
                pos_sum += regret;
            }
        }

        if pos_sum > 0.0 {
            // Use regret matching
            regrets_with_edges
                .into_iter()
                .map(|(edge, r)| Decision::from((edge, if r > 0.0 { r / pos_sum } else { 0.0 })))
                .collect()
        } else {
            // Fall back to accumulated policy σ̄ - get all policies in one batch
            let all_policies = self.profile.get_all_policies(info);
            
            let mut sum_policy = 0.0f32;
            let mut policies_with_edges = Vec::with_capacity(all_choices.len());
            
            for choice_edge in all_choices.iter() {
                let policy = all_policies
                    .iter()
                    .find_map(|(e, p)| if e == choice_edge { Some(*p) } else { None })
                    .unwrap_or(0.0);
                
                policies_with_edges.push((*choice_edge, policy));
                sum_policy += policy;
            }
                
            if sum_policy > 0.0 {
                policies_with_edges
                    .into_iter()
                    .map(|(edge, w)| Decision::from((edge, w / sum_policy)))
                    .collect()
            } else if !self.warm_start.is_empty() {
                self.warm_start.clone()
            } else {
                // Final fallback to uniform
                let n = all_choices.len() as f32;
                all_choices
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