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
                        } else {
                            // Last resort uniform
                            let n = info.choices().len() as f32;
                            info.choices()
                                .into_iter()
                                .map(|edge| Decision::from((edge, 1.0 / n)))
                                .collect()
                        };
                        log::debug!("Using average-policy fallback strategy (shift sum zero)");
                        strategy
                    }
                }
            }
            None => {
                log::debug!("Using warm-start strategy (no root info)");
                self.warm_start.clone()
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