use super::{Edge, Game, Info, Turn, Policy};
use crate::analysis::response::Decision;
use crate::mccfr::traits::profile::Profile;
use rustc_hash::FxHashMap;
use rand::{SeedableRng, rngs::SmallRng};

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
}

impl SubgameProfile {
    pub fn new(warm_start: Vec<Decision>) -> Self {
        Self {
            iterations: 0,
            regrets: FxHashMap::default(),
            policies: FxHashMap::default(),
            warm_start,
        }
    }

    /// Get the final strategy for the root node
    pub fn get_strategy_for_root(&self) -> Vec<Decision> {
        // Find the root info (the one with the most entries)
        let root_info = self.policies
            .keys()
            .map(|(info, _)| info)
            .max_by_key(|info| {
                self.policies.iter()
                    .filter(|((i, _), _)| i == *info)
                    .count()
            });

        match root_info {
            Some(info) => {
                // Get all edges for this info and normalize
                let edges: Vec<(Edge, f32)> = self.policies
                    .iter()
                    .filter(|((i, _), _)| i == info)
                    .map(|((_, e), &w)| (e.clone(), w.max(crate::POLICY_MIN)))
                    .collect();
                
                let total: f32 = edges.iter().map(|(_, w)| w).sum();
                
                if total > 0.0 {
                    edges.into_iter()
                        .map(|(edge, weight)| Decision::from((edge, weight / total)))
                        .collect()
                } else {
                    self.warm_start.clone()
                }
            }
            None => self.warm_start.clone(),
        }
    }

    /// Apply updates from a batch of counterfactuals
    pub fn apply_updates(&mut self, updates: Vec<(Info, Policy<Edge>, Policy<Edge>)>) {
        for (info, regret_vec, policy_vec) in updates {
            // Apply regret updates
            for (edge, regret) in regret_vec {
                let key = (info.clone(), edge);
                self.regrets.entry(key)
                    .and_modify(|r| *r = (*r + regret).clamp(crate::REGRET_MIN, crate::REGRET_MAX))
                    .or_insert(regret);
            }
            
            // Apply policy updates
            for (edge, policy) in policy_vec {
                let key = (info.clone(), edge);
                self.policies.entry(key)
                    .and_modify(|p| *p = (*p + policy).max(crate::POLICY_MIN))
                    .or_insert(policy);
            }
        }
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
        Turn::Choice(self.iterations % crate::N)
    }

    fn epochs(&self) -> usize {
        self.iterations
    }

    fn sum_policy(&self, info: &Self::I, edge: &Self::E) -> crate::Probability {
        self.policies
            .get(&(info.clone(), edge.clone()))
            .copied()
            .unwrap_or(0.0)
    }

    fn sum_regret(&self, info: &Self::I, edge: &Self::E) -> crate::Utility {
        self.regrets
            .get(&(info.clone(), edge.clone()))
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
} 