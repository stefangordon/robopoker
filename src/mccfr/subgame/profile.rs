use super::{Edge, Game, Info, Turn, Policy};
use crate::analysis::response::Decision;
use crate::mccfr::traits::profile::Profile;
use crate::mccfr::traits::info::Info as InfoTrait;
use crate::mccfr::structs::infoset::InfoSet;
use rustc_hash::FxHashMap;
use rand::{SeedableRng, rngs::SmallRng};
use crate::mccfr::subgame::SubgameParams;
use crate::mccfr::structs::node::Node;
use crate::mccfr::traits::game::Game as GameTrait;

/// Lightweight profile for subgame solving
pub struct SubgameProfile {
    /// Current iteration count
    iterations: usize,
    /// Accumulated regrets (info, edge) -> regret
    regrets: FxHashMap<(Info, Edge), f32>,
    /// Accumulated policy weights (info, edge) -> weight
    policies: FxHashMap<(Info, Edge), f32>,
    /// Warm-start strategy from blueprint
    warm_start: Vec<Decision>,
    /// Which player we are updating (walker)
    walker_turn: Turn,
    /// Root info for strategy extraction
    root_info: Option<Info>,
    /// numeric parameters for subgame solver
    params: SubgameParams,
}

impl SubgameProfile {
    pub fn new(warm_start: Vec<Decision>, walker_turn: Turn) -> Self {
        Self {
            iterations: 0,
            regrets: FxHashMap::default(),
            policies: FxHashMap::default(),
            warm_start,
            walker_turn,
            root_info: None,
            params: SubgameParams::default(),
        }
    }

    /// Set the root info for strategy extraction
    pub fn set_root_info(&mut self, info: Info) {
        self.root_info = Some(info);

        // Initialize regrets and policies from warm start if available
        if !self.warm_start.is_empty() {
            self.initialize_from_warm_start(&info);
        }
    }

        /// Initialize regrets and policies from warm start strategy
    fn initialize_from_warm_start(&mut self, info: &Info) {
        let canonical_info = Self::canonical(info);
        
        // Debug: log the warm start strategy
        if log::log_enabled!(log::Level::Debug) {
            log::debug!("Warm start strategy contains {} actions:", self.warm_start.len());
            for (i, decision) in self.warm_start.iter().enumerate() {
                log::debug!("  {}: edge={:?} weight={:.6}", i, decision.edge(), decision.weight());
            }
        }
        
        // Get the subgame's available edges
        let subgame_edges = info.choices();
        
        // Debug: log the subgame edges
        if log::log_enabled!(log::Level::Debug) {
            log::debug!("Subgame has {} available edges:", subgame_edges.len());
            for (i, edge) in subgame_edges.iter().enumerate() {
                log::debug!("  {}: {:?}", i, edge);
            }
        }
        
        // Translate warm start strategy to subgame action space
        let translated_strategy = self.translate_warm_start_to_subgame(subgame_edges);
        
        if !translated_strategy.is_empty() {
            let total_weight: f32 = translated_strategy.iter().map(|(_, w)| *w).sum();
            let translated_count = translated_strategy.len();

            if total_weight > 0.0 {
                for (edge, weight) in translated_strategy {
                    let normalized_weight = weight / total_weight;

                    // Initialize with positive policy weight to seed the strategy
                    self.policies.insert((canonical_info, edge), normalized_weight * 10.0);

                    // Initialize regrets to zero (neutral starting point)
                    self.regrets.insert((canonical_info, edge), 0.0);
                }

                log::debug!("Translated and initialized subgame profile from warm start: {} -> {} actions",
                           self.warm_start.len(), translated_count);
            }
        } else {
            log::debug!("Could not translate warm start strategy to subgame action space");
        }
    }

        /// Translate warm start strategy from blueprint action space to subgame action space
    fn translate_warm_start_to_subgame(&self, subgame_edges: Vec<Edge>) -> Vec<(Edge, f32)> {
        let mut translated = Vec::new();

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

        // Count total raise edges first (before consuming the vector)
        let raise_edge_count = subgame_edges.iter()
            .filter(|e| matches!(e, Edge::Raise(_)))
            .count() as f32;

        // Map to subgame edges
        for edge in subgame_edges {
            let weight = match edge {
                Edge::Check => check_weight,
                Edge::Fold => fold_weight,
                Edge::Call => call_weight,
                Edge::Shove => shove_weight,
                Edge::Draw => draw_weight,
                                Edge::Raise(odds) => {
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
                    
                    if raise_edge_count > 0.0 {
                        if raise_weight > 0.0 {
                            // Distribute actual raise weight from blueprint
                            (raise_weight * preference) / raise_edge_count
                        } else {
                            // Blueprint has no raises, but give subgame raises small weight
                            // Use a fraction of the total non-raise weight
                            let total_non_raise_weight = check_weight + fold_weight + call_weight + shove_weight + draw_weight;
                            let fallback_raise_weight = total_non_raise_weight * 0.1; // 10% of other actions
                            (fallback_raise_weight * preference) / raise_edge_count
                        }
                    } else {
                        0.0
                    }
                }
            };

            if weight > 0.0 {
                translated.push((edge, weight));
            }
        }

        translated
    }

    /// Canonicalize an Info by stripping its history so that
    /// all nodes representing the same (present,futures) bucket
    /// share the same entry in the regrets / policies tables.
    fn canonical(info: &Info) -> Info {
        use crate::gameplay::path::Path;
        Info::from((Path::default(), *info.present(), info.futures().clone()))
    }

    /// Get the final strategy for the root node
    pub fn get_strategy_for_root(&self) -> Vec<Decision> {
        match &self.root_info {
            Some(info) => {
                // Get all edges for this info and normalize using regret-matching
                if log::log_enabled!(log::Level::Debug) {
                    for edge in info.choices() {
                        let r = self.sum_regret(info, &edge);
                        let p = self.sum_policy(info, &edge);
                        log::debug!("Root info debug – edge {:?}: regret={:.6} policy={:.6}", edge, r, p);
                    }
                }
                let edges: Vec<(Edge, f32)> = info.choices()
                    .into_iter()
                    .map(|edge| {
                        let regret = self.sum_regret(info, &edge);
                        (edge.clone(), regret)
                    })
                    .collect();

                // Compute positive portion sum for regret matching
                let pos_sum: f32 = edges
                    .iter()
                    .map(|(_, r)| if *r > 0.0 { *r } else { 0.0 })
                    .sum();

                log::trace!("Strategy extraction: {} edges, pos_sum={:.6}", edges.len(), pos_sum);
                for (edge, regret) in &edges {
                    log::trace!("  Edge {:?}: regret={:.6}", edge, regret);
                }

                if pos_sum > 0.0 {
                    // Standard regret-matching on positive regrets
                    let strategy = edges.into_iter()
                        .map(|(edge, r)| {
                            let w = if r > 0.0 { r / pos_sum } else { 0.0 };
                            Decision::from((edge, w))
                        })
                        .collect();
                    log::debug!("Using regret-matching strategy");
                    strategy
                } else {
                    // Shifted regret fallback (adds a constant so one edge becomes zero)
                    let min_r = edges.iter().map(|(_, r)| *r).fold(f32::INFINITY, f32::min);
                    let shifted: Vec<(Edge, f32)> = edges
                        .iter()
                        .map(|(e, r)| (e.clone(), r - min_r))
                        .collect();
                    let shift_sum: f32 = shifted.iter().map(|(_, w)| *w).sum();

                    if shift_sum > 0.0 {
                        let strategy = shifted
                            .into_iter()
                            .map(|(edge, w)| Decision::from((edge, w / shift_sum)))
                            .collect();
                        log::debug!("Using shifted-regret strategy fallback");
                        strategy
                    } else {
                        // Fallback to average accumulated policy σ̄
                        let weights: Vec<(Edge, f32)> = info
                            .choices()
                            .into_iter()
                            .map(|edge| {
                                let w = self.sum_policy(info, &edge);
                                (edge, w)
                            })
                            .collect();
                        let sum_policy: f32 = weights.iter().map(|(_, w)| *w).sum();
                        let strategy = if sum_policy > 0.0 {
                            weights
                                .into_iter()
                                .map(|(edge, w)| Decision::from((edge, w / sum_policy)))
                                .collect()
                        } else if !self.warm_start.is_empty() {
                            // Use warm start strategy if available
                            log::debug!("Using warm-start strategy (no accumulated data)");
                            self.warm_start.clone()
                        } else {
                            // Last resort uniform
                            let n = info.choices().len() as f32;
                            let strategy = info.choices()
                                .into_iter()
                                .map(|edge| Decision::from((edge, 1.0 / n)))
                                .collect();
                            log::debug!("Using uniform fallback strategy (no warm start available)");
                            strategy
                        };
                        strategy
                    }
                }
            }
            None => {
                if !self.warm_start.is_empty() {
                    log::debug!("Using warm-start strategy (no root info)");
                    self.warm_start.clone()
                } else {
                    log::debug!("No warm start or root info available, returning empty strategy");
                    vec![]
                }
            }
        }
    }

    /// Apply updates from a batch of counterfactuals
    pub fn apply_updates(&mut self, updates: Vec<(Info, Policy<Edge>, Policy<Edge>)>) {
        let mut regret_count = 0;
        let mut total_regret = 0.0;

        for (info, regret_vec, policy_vec) in updates {
            // Apply regret updates
            for (edge, regret) in regret_vec {
                let key = (Self::canonical(&info), edge);
                self.regrets.entry(key)
                    .and_modify(|r| *r = (*r + regret).clamp(self.params.regret_min, self.params.regret_max))
                    .or_insert(regret);
                regret_count += 1;
                total_regret += regret.abs();
            }

            // Apply policy updates (weighted by current iteration for averaging)
            for (edge, policy) in policy_vec {
                let key = (Self::canonical(&info), edge);
                let weight = policy * (self.iterations as f32 + 1.0);
                self.policies.entry(key)
                    .and_modify(|p| *p += weight)
                    .or_insert(weight);
            }
        }

        if regret_count > 0 {
            log::trace!("Applied {} regret updates, avg magnitude: {:.6}",
                       regret_count, total_regret / regret_count as f32);
        }
    }

    /// Compute regret vector for an infoset (delegating to trait implementation)
    pub fn regret_vector(&self, infoset: &InfoSet<Turn, Edge, Game, Info>) -> Policy<Edge> {
        Profile::regret_vector(self, infoset)
    }

    /// Compute policy vector for an infoset (delegating to trait implementation)
    pub fn policy_vector(&self, infoset: &InfoSet<Turn, Edge, Game, Info>) -> Policy<Edge> {
        Profile::policy_vector(self, infoset)
    }
}

impl Profile for SubgameProfile {
    type T = Turn;
    type E = Edge;
    type G = Game;
    type I = Info;

    fn increment(&mut self) {
        self.iterations += 1;
    }

    fn walker(&self) -> Self::T {
        self.walker_turn
    }

    fn epochs(&self) -> usize {
        self.iterations
    }

    fn sum_policy(&self, info: &Self::I, edge: &Self::E) -> crate::Probability {
        self.policies
            .get(&(Self::canonical(&info), edge.clone()))
            .copied()
            .unwrap_or(0.0)
    }

    fn sum_regret(&self, info: &Self::I, edge: &Self::E) -> crate::Utility {
        self.regrets
            .get(&(Self::canonical(&info), edge.clone()))
            .copied()
            .unwrap_or(0.0)
    }

    // Use more aggressive parameters for faster convergence in subgames
    fn threshold(&self) -> crate::Entropy {
        0.5 // Lower than blueprint for more focused sampling
    }

    fn activation(&self) -> crate::Energy {
        0.1 // Small activation for stability
    }

    fn exploration(&self) -> crate::Probability {
        0.05 // Higher exploration in subgames
    }

    fn rng(&self, _info: &Self::I) -> SmallRng {
        SmallRng::seed_from_u64(self.iterations as u64)
    }

    fn relative_value(
        &self,
        root: &Node<Self::T, Self::E, Self::G, Self::I>,
        leaf: &Node<Self::T, Self::E, Self::G, Self::I>,
    ) -> crate::Utility {
        let reach = self.relative_reach(root, leaf);
        let payoff = leaf.game().payoff(root.game().turn());
        let sampling = self.sampling_reach(leaf).max(self.params.sampling_eps);
        let mut result = reach * payoff / sampling;
        if !result.is_finite() {
            result = result.signum() * self.params.regret_max;
        }
        result
    }
}