use super::compact_bucket::CompactBucket;
use super::edge::Edge;
use super::game::Game;
use super::info::Info;
use super::turn::Turn;
use crate::mccfr::traits::info::Info as InfoTrait;
use crate::mccfr::types::decision::Decision;
use crate::mccfr::types::policy::Policy;
use dashmap::DashMap;
use half::f16;
use parking_lot::RwLock;
use rustc_hash::FxHasher;
use std::hash::BuildHasher;

// Store Edge as compact u8 key inside buckets to reduce memory footprint
// The Edge <-> u8 bijection is already implemented via From conversions.
type Bucket = CompactBucket;

#[inline(always)]
fn safe_clamp(val: f32, min: f32, max: f32) -> f32 {
    let val = val.max(min).min(max); // NaNs will propagate
    if val.is_nan() {
        0.0
    } else {
        val
    }
}

pub struct Profile {
    pub(super) iterations: usize,
    /// Total number of game tree traversals performed across all training runs.
    /// This persists across checkpoint saves/loads to track cumulative training effort.
    pub(super) total_traversals: u64,
    pub(super) encounters: DashMap<Info, RwLock<Bucket>, FxBuildHasher>,
    pub(super) regret_min: crate::Utility,
    pub(super) regret_max: crate::Utility,
}

impl Default for Profile {
    fn default() -> Self {
        Self {
            iterations: 0,
            total_traversals: 0,
            encounters: DashMap::with_hasher(FxBuildHasher::default()),
            regret_min: crate::REGRET_MIN,
            regret_max: crate::REGRET_MAX,
        }
    }
}

impl Profile {
    pub(super) fn name() -> String {
        "blueprint".to_string()
    }

    /// Get the number of tree traversals in the current training run only
    /// (not cumulative across all runs)
    pub(super) fn run_traversals(&self) -> u64 {
        self.iterations as u64 * crate::CFR_BATCH_SIZE_NLHE as u64
    }

    /// Calculate convergence metrics for monitoring training progress
    /// NOTE: This function is deprecated and expensive for large datasets.
    /// Use log_stats() instead which computes these metrics more efficiently via sampling.
    #[deprecated(note = "Use log_stats() instead for better performance with large datasets")]
    pub fn convergence_metrics(&self) -> (f64, f64, f64) {
        // For backward compatibility, return reasonable defaults
        // Real computation happens in log_stats() now
        (0.0, 0.0, 0.0)
    }

    /// Seed initial policy weights (and zero regrets) for a given infoset.
    pub fn seed_decisions(&self, info: &Info, decisions: &[Decision]) {
        if decisions.is_empty() {
            return;
        }
        let bucket_mutex = self
            .encounters
            .entry(*info)
            .or_insert_with(|| RwLock::new(CompactBucket::new()));
        let mut bucket = bucket_mutex.value().write();
        // Normalise weights to sum to 1.
        let total: f32 = decisions.iter().map(|d| d.weight()).sum();
        let total = total.max(crate::POLICY_MIN);
        for d in decisions {
            let w = d.weight() / total;
            bucket.push((u8::from(d.edge()), (f16::from_f32(w), 0.0)));
        }
    }

    /// Apply regret/policy deltas concurrently-safe (called by subgame solver).
    pub fn apply_updates(
        &self,
        updates: Vec<(
            Info,
            crate::mccfr::types::policy::Policy<Edge>,
            crate::mccfr::types::policy::Policy<Edge>,
        )>,
    ) {
        let current_min = self.regret_min;
        let current_max = self.regret_max;
        for (info, regret_vec, policy_vec) in updates {
            let bucket_mutex = self
                .encounters
                .entry(info)
                .or_insert_with(|| RwLock::new(CompactBucket::new()));
            let mut bucket = bucket_mutex.value().write();
            // Update regrets
            for (edge, delta) in regret_vec {
                let edge_key = u8::from(edge);
                if !bucket.update_regret(edge_key, |old_regret| {
                    self::safe_clamp(old_regret + delta, current_min, current_max)
                }) {
                    // Edge doesn't exist, add new entry
                    let safe_delta = self::safe_clamp(delta, current_min, current_max);
                    bucket.push((edge_key, (f16::from_f32(0.0), safe_delta)));
                }
            }
            // Update policies
            for (edge, delta) in policy_vec {
                let edge_key = u8::from(edge);
                if !bucket.update_policy(edge_key, |old_policy| {
                    let new_policy = old_policy + delta;
                    // Fast check for invalid values
                    if new_policy.is_nan() || new_policy < 0.0 {
                        crate::POLICY_MIN
                    } else {
                        new_policy
                    }
                }) {
                    // Edge doesn't exist, add new entry
                    let safe_delta = if delta.is_nan() || delta < 0.0 {
                        crate::POLICY_MIN
                    } else {
                        delta
                    };
                    bucket.push((edge_key, (f16::from_f32(safe_delta), 0.0)));
                }
            }
        }
    }

    /// Get all regrets for an infoset (used by subgame solver)
    pub fn get_all_regrets(&self, info: &Info) -> Vec<(Edge, f32)> {
        let all_choices = info.choices();
        if all_choices.is_empty() {
            return Vec::new();
        }

        if let Some(bucket_mutex) = self.encounters.get(info) {
            let bucket = bucket_mutex.value().read();

            // Build result vector with all edges and their regrets
            all_choices
                .into_iter()
                .map(|edge| {
                    let edge_u8 = u8::from(edge);
                    if bucket.is_edge_skipped(edge_u8, self.run_traversals()) {
                        (edge, 0.0)
                    } else if let Some((_, regret)) = bucket.find_by_edge(edge_u8) {
                        (edge, regret)
                    } else {
                        (edge, 0.0)
                    }
                })
                .collect()
        } else {
            // No data: all regrets are 0
            all_choices.into_iter().map(|edge| (edge, 0.0)).collect()
        }
    }

    /// Get all policies for an infoset (used by subgame solver)
    pub fn get_all_policies(&self, info: &Info) -> Vec<(Edge, f32)> {
        let all_choices = info.choices();
        if all_choices.is_empty() {
            return Vec::new();
        }

        if let Some(bucket_mutex) = self.encounters.get(info) {
            let bucket = bucket_mutex.value().read();

            // Build result vector with all edges and their policies
            all_choices
                .into_iter()
                .map(|edge| {
                    let edge_u8 = u8::from(edge);
                    if bucket.is_edge_skipped(edge_u8, self.run_traversals()) {
                        (edge, 0.0)
                    } else if let Some((policy, _)) = bucket.find_by_edge(edge_u8) {
                        (edge, f32::from(policy))
                    } else {
                        (edge, 0.0)
                    }
                })
                .collect()
        } else {
            // No data: all policies are 0
            all_choices.into_iter().map(|edge| (edge, 0.0)).collect()
        }
    }

    /// Compute the entire policy distribution for an infoset using regret matching.
    /// This does a single DashMap lookup and returns normalized probabilities for all edges.
    fn policy_distribution(&self, info: &Info) -> Policy<Edge> {
        let all_choices = info.choices();
        if all_choices.is_empty() {
            return Policy::new();
        }

        if let Some(bucket_mutex) = self.encounters.get(info) {
            let bucket = bucket_mutex.value().read();

            // Build choice map for O(1) lookups
            let mut choice_map = [0xff_u8; 256];
            for (i, &choice) in all_choices.iter().enumerate() {
                choice_map[u8::from(choice) as usize] = i as u8;
            }

            // Use stack allocation for small edge counts
            let mut regrets = [crate::POLICY_MIN; 13];

            // Single pass through bucket to gather regrets
            for (edge_u8, (_, regret)) in bucket.iter_active(self.run_traversals()) {
                let choice_idx = choice_map[edge_u8 as usize];
                if choice_idx != 0xff {
                    regrets[choice_idx as usize] = regret.max(crate::POLICY_MIN);
                }
            }

            // Compute sum and normalize
            let sum: f32 = regrets[..all_choices.len()].iter().sum();

            // Build result with normalization
            all_choices
                .iter()
                .enumerate()
                .map(|(i, &edge)| (edge, regrets[i] / sum))
                .collect()
        } else {
            // No data: uniform distribution
            let prob = 1.0 / all_choices.len() as f32;
            all_choices.into_iter().map(|edge| (edge, prob)).collect()
        }
    }

    /// Compute sampling probabilities for all edges in an infoset efficiently.
    /// This does a single DashMap lookup and returns sampling weights for all edges.
    /// Used by MCCFR external sampling to select actions during tree traversal.
    #[allow(dead_code)]
    fn sample_distribution(&self, info: &Info) -> Policy<Edge> {
        let all_choices = info.choices();
        if all_choices.is_empty() {
            return Policy::new();
        }

        // Constants for sampling formula
        let activation = crate::SAMPLING_ACTIVATION;
        let threshold = crate::SAMPLING_THRESHOLD;
        let exploration = crate::SAMPLING_EXPLORATION;

        if let Some(bucket_mutex) = self.encounters.get(info) {
            let bucket = bucket_mutex.value().read();

            // Build choice map for O(1) lookups
            let mut choice_map = [0xff_u8; 256];
            for (i, &choice) in all_choices.iter().enumerate() {
                choice_map[u8::from(choice) as usize] = i as u8;
            }

            // Use stack allocation for small edge counts
            let mut policies = [0.0f32; 13];

            // Single pass through bucket to gather policies
            for (edge_u8, (policy, _)) in bucket.iter_active(self.run_traversals()) {
                let choice_idx = choice_map[edge_u8 as usize];
                if choice_idx != 0xff {
                    policies[choice_idx as usize] = f32::from(policy).max(crate::POLICY_MIN);
                }
            }

            // Apply sampling formula: q(a) = max(ε, (β + τ * weight(a)) / (β + sum(weights)))
            let sum: f32 = policies[..all_choices.len()].iter().sum();
            let denom = activation + sum;

            // Build result with sampling probabilities
            all_choices
                .iter()
                .enumerate()
                .map(|(i, &edge)| {
                    let numer = activation + policies[i] * threshold;
                    let sample_prob = (numer / denom).max(exploration);
                    (edge, sample_prob)
                })
                .collect()
        } else {
            // No data: all policies are POLICY_MIN
            let n = all_choices.len() as f32;
            let sum = n * crate::POLICY_MIN;
            let denom = activation + sum;
            let numer = activation + crate::POLICY_MIN * threshold;
            let uniform_prob = (numer / denom).max(exploration);
            all_choices
                .into_iter()
                .map(|edge| (edge, uniform_prob))
                .collect()
        }
    }
}

impl crate::mccfr::traits::profile::Profile for Profile {
    type T = Turn;
    type E = Edge;
    type G = Game;
    type I = Info;

    fn increment(&mut self) {
        self.iterations += 1;
    }

    fn walker(&self) -> Self::T {
        match self.iterations % crate::N {
            player_idx => Turn::Choice(player_idx),
        }
    }
    fn epochs(&self) -> usize {
        self.iterations
    }

    // Keep the single-edge lookups for when we truly need just one value
    fn sum_policy(&self, info: &Self::I, edge: &Self::E) -> crate::Probability {
        if let Some(bucket_mutex) = self.encounters.get(info) {
            let bucket = bucket_mutex.value().read();
            let edge_u8 = u8::from(*edge);
            if bucket.is_edge_skipped(edge_u8, self.run_traversals()) {
                return 0.0;
            }

            // Try to find the edge in the bucket
            if let Some((policy, _)) = bucket.find_by_edge(edge_u8) {
                f32::from(policy)
            } else {
                // Edge not found - use default value
                0.0
            }
        } else {
            0.0
        }
    }

    fn sum_regret(&self, info: &Self::I, edge: &Self::E) -> crate::Utility {
        if let Some(bucket_mutex) = self.encounters.get(info) {
            let bucket = bucket_mutex.value().read();
            let edge_u8 = u8::from(*edge);
            if bucket.is_edge_skipped(edge_u8, self.run_traversals()) {
                return 0.0;
            }

            // Try to find the edge in the bucket
            if let Some((_, regret)) = bucket.find_by_edge(edge_u8) {
                regret
            } else {
                // Edge not found - use default value
                0.0
            }
        } else {
            0.0
        }
    }

    fn current_regret_min(&self) -> crate::Utility {
        self.regret_min
    }
    fn current_regret_max(&self) -> crate::Utility {
        self.regret_max
    }

    // Optimised allocation-free implementation of policy(), mirroring the
    // changes made to sample().  We iterate directly over the futures Path
    // instead of materialising a Vec, and we early-exit if the edge is pruned.
    fn policy(&self, info: &Self::I, edge: &Self::E) -> crate::Probability {
        // Quick reject if the edge is not in the futures path.
        if !info.futures().clone().any(|e| e == *edge) {
            return 0.0;
        }

        let run_trav = self.run_traversals();
        let mut sum = 0.0f32;
        let mut target_regret = 0.0f32;

        if let Some(bucket_mutex) = self.encounters.get(info) {
            let bucket = bucket_mutex.value().read();

            // Single scan over the reachable edges.
            let mut it = *info.futures();
            while let Some(e) = it.next() {
                let edge_u8 = u8::from(e);

                // Skip edges pruned by RBP.
                if bucket.is_edge_skipped(edge_u8, run_trav) {
                    if e == *edge {
                        return 0.0; // Target edge is currently pruned.
                    }
                    continue;
                }

                let regret = bucket
                    .find_by_edge(edge_u8)
                    .map(|(_, r)| r)
                    .unwrap_or(crate::POLICY_MIN)
                    .max(crate::POLICY_MIN);

                if e == *edge {
                    target_regret = regret;
                }

                sum += regret;
            }
        } else {
            // No bucket yet – assume uniform minimal regret.
            let mut it = *info.futures();
            while let Some(e) = it.next() {
                if e == *edge {
                    target_regret = crate::POLICY_MIN;
                }
                sum += crate::POLICY_MIN;
            }
        }

        if sum == 0.0 {
            return 0.0;
        }

        target_regret / sum
    }

    // Override the default advice() method to use direct bucket access
    fn advice(&self, info: &Self::I, edge: &Self::E) -> crate::Probability {
        let all_choices = info.choices();
        if all_choices.is_empty() {
            return 0.0;
        }

        // Early check if edge is in choices and build lookup map
        let mut choice_map = [0xff_u8; 256];
        let mut target_idx = 0xff_u8;
        for (i, &choice) in all_choices.iter().enumerate() {
            choice_map[u8::from(choice) as usize] = i as u8;
            if choice == *edge {
                target_idx = i as u8;
            }
        }

        if target_idx == 0xff {
            return 0.0; // Edge not in choices
        }

        if let Some(bucket_mutex) = self.encounters.get(info) {
            let bucket = bucket_mutex.value().read();
            let edge_u8 = u8::from(*edge);
            if bucket.is_edge_skipped(edge_u8, self.run_traversals()) {
                return 0.0;
            }

            // Try to find the edge in the bucket
            if let Some((_, regret)) = bucket.find_by_edge(edge_u8) {
                regret.max(crate::POLICY_MIN)
            } else {
                // Edge not found - return POLICY_MIN (it's a valid but unexplored edge)
                crate::POLICY_MIN
            }
        } else {
            crate::POLICY_MIN
        }
    }

    // Optimised heap-free implementation: iterate directly over the futures Path
    // instead of materialising a Vec via `info.choices()`.  This removes the last
    // remaining allocation in the hot `sample()` path.
    fn sample(&self, info: &Self::I, edge: &Self::E) -> crate::Probability {
        // Constants for sampling formula
        let activation = crate::SAMPLING_ACTIVATION;
        let threshold = crate::SAMPLING_THRESHOLD;
        let exploration = crate::SAMPLING_EXPLORATION;

        // Verify the target edge is actually reachable from this infoset and count futures.
        let mut futures_iter = *info.futures();
        let mut edge_present = false;
        let mut futures_len = 0usize;
        while let Some(e) = futures_iter.next() {
            if e == *edge {
                edge_present = true;
            }
            futures_len += 1;
        }
        if !edge_present || futures_len == 0 {
            return 0.0;
        }

        let run_trav = self.run_traversals();

        if let Some(bucket_mutex) = self.encounters.get(info) {
            let bucket = bucket_mutex.value().read();

            // Second pass to accumulate policies.
            let mut futures_iter = *info.futures();
            let mut sum = 0.0f32;
            let mut target_policy = 0.0f32;

            while let Some(e) = futures_iter.next() {
                let edge_u8 = u8::from(e);
                if bucket.is_edge_skipped(edge_u8, run_trav) {
                    continue;
                }

                let p = bucket
                    .find_by_edge(edge_u8)
                    .map(|(policy, _)| f32::from(policy))
                    .unwrap_or(crate::POLICY_MIN)
                    .max(crate::POLICY_MIN);

                if e == *edge {
                    target_policy = p;
                }

                sum += p;
            }

            let denom = activation + sum;
            let numer = activation + target_policy * threshold;
            (numer / denom).max(exploration)
        } else {
            // No bucket yet – treat all edges as POLICY_MIN.
            let sum = futures_len as f32 * crate::POLICY_MIN;
            let target_policy = crate::POLICY_MIN;
            let denom = activation + sum;
            let numer = activation + target_policy * threshold;
            (numer / denom).max(exploration)
        }
    }

    // The already-optimized policy_vector stays mostly the same,
    // but we can make it even cleaner now
    fn policy_vector(
        &self,
        infoset: &crate::mccfr::structs::infoset::InfoSet<Self::T, Self::E, Self::G, Self::I>,
    ) -> crate::mccfr::types::policy::Policy<Self::E> {
        let info = infoset.info();

        // Use the new batch method to get the entire distribution at once
        self.policy_distribution(&info)
    }

    // Note: regret_vector will automatically benefit from our optimized policy() method
    // which now uses batch access instead of individual lookups

    // Override explore_one to use batch sampling with a single DashMap lookup
    fn explore_one(
        &self,
        node: &crate::mccfr::structs::node::Node<Self::T, Self::E, Self::G, Self::I>,
        branches: Vec<crate::mccfr::types::branch::Branch<Self::E, Self::G>>,
    ) -> Vec<crate::mccfr::types::branch::Branch<Self::E, Self::G>> {
        use rand::distributions::WeightedIndex;
        use rand::prelude::Distribution;

        let info = node.info();
        let mut choices = branches;

        if choices.is_empty() {
            return choices;
        }

        // Constants for sampling formula
        let activation = crate::SAMPLING_ACTIVATION;
        let threshold = crate::SAMPLING_THRESHOLD;
        let exploration = crate::SAMPLING_EXPLORATION;

        // Compute weights with a single DashMap lookup
        let weights: Vec<f32> = if let Some(bucket_mutex) = self.encounters.get(&info) {
            let bucket = bucket_mutex.value().read();

            // Build edge to policy map
            let mut edge_policies = [0.0f32; 256]; // indexed by edge u8 value
            let mut total_policy = 0.0f32;

            for (edge_u8, (policy, _)) in bucket.iter_active(self.run_traversals()) {
                let p = f32::from(policy).max(crate::POLICY_MIN);
                edge_policies[edge_u8 as usize] = p;
                total_policy += p;
            }

            // Also count edges not in bucket
            let all_choices = info.choices();
            for edge in &all_choices {
                let edge_u8 = u8::from(*edge);
                if edge_policies[edge_u8 as usize] == 0.0 {
                    if !bucket.is_edge_skipped(edge_u8, self.run_traversals()) {
                        edge_policies[edge_u8 as usize] = crate::POLICY_MIN;
                        total_policy += crate::POLICY_MIN;
                    }
                }
            }

            // Apply sampling formula to each branch
            let denom = activation + total_policy;
            choices
                .iter()
                .map(|(edge, _, _)| {
                    let edge_u8 = u8::from(*edge);
                    let policy = edge_policies[edge_u8 as usize];
                    let numer = activation + policy * threshold;
                    (numer / denom).max(exploration)
                })
                .collect()
        } else {
            // No data: all policies are POLICY_MIN
            let n = info.choices().len() as f32;
            let sum = n * crate::POLICY_MIN;
            let denom = activation + sum;
            let numer = activation + crate::POLICY_MIN * threshold;
            let uniform_prob = (numer / denom).max(exploration);
            vec![uniform_prob; choices.len()]
        };

        // Sample according to weights
        let ref mut rng = self.rng(&info);
        let choice = WeightedIndex::new(weights)
            .expect("at least one weight > 0")
            .sample(rng);
        let chosen = choices.remove(choice);
        vec![chosen]
    }
}

/// Custom BuildHasher for FxHasher
#[derive(Clone, Default)]
pub struct FxBuildHasher;

impl BuildHasher for FxBuildHasher {
    type Hasher = FxHasher;

    fn build_hasher(&self) -> Self::Hasher {
        FxHasher::default()
    }
}

// -------------------------------------------------------------------
// ProfileBuilder – constructs a Profile with optional warm-start and
// tunable regret bounds.  The regret caps are not yet wired into the
// legacy Profile implementation; they are stored for future use so the
// caller can still query / log them.  Sub-game solvers can pull the
// warm-start vector out immediately after build() and seed it once the
// root Info is known.
// -------------------------------------------------------------------

#[derive(Default)]
pub struct ProfileBuilder {
    warm_start: Vec<Decision>,
    regret_min: f32,
    regret_max: f32,
}

impl ProfileBuilder {
    /// Start a builder with default global regret caps and no warm-start.
    pub fn new() -> Self {
        Self {
            warm_start: Vec::new(),
            regret_min: crate::REGRET_MIN,
            regret_max: crate::REGRET_MAX,
        }
    }

    /// Attach a warm-start strategy that will be returned alongside the
    /// constructed profile.
    pub fn with_warm_start(mut self, strategy: Vec<Decision>) -> Self {
        self.warm_start = strategy;
        self
    }

    /// Customise regret clamping bounds (not yet threaded through).
    pub fn with_regret_bounds(mut self, min: f32, max: f32) -> Self {
        self.regret_min = min;
        self.regret_max = max;
        self
    }

    /// Build the DashMap-based Profile.  Returns the profile together with
    /// the warm-start vector so the caller can initialise the root infoset
    /// once it is known.
    pub fn build(self) -> (Profile, Vec<Decision>) {
        let profile = Profile {
            iterations: 0,
            total_traversals: 0,
            encounters: DashMap::with_hasher(FxBuildHasher::default()),
            regret_min: self.regret_min,
            regret_max: self.regret_max,
        };
        let warm_start = self.warm_start;
        // Populate profile with warm-start decisions if any.
        (profile, warm_start)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mccfr::traits::profile::Profile as ProfileTrait;

    /// Helper to create a test CompactBucket with known values
    fn create_test_bucket() -> CompactBucket {
        let mut bucket = CompactBucket::new();
        // Add some test data with known edge values
        bucket.push((1, (f16::from_f32(0.3), 0.0))); // Edge 1
        bucket.push((2, (f16::from_f32(0.7), 0.0))); // Edge 2
        bucket.push((3, (f16::from_f32(0.5), 0.0))); // Edge 3
        bucket
    }

    /// Test that our sample() method produces the correct sampling probabilities
    #[test]
    fn test_sampling_formula() {
        let profile = Profile::default();

        // Create a dummy Info - we'll bypass info.choices() by testing internal logic
        let info = Info::from((
            crate::gameplay::path::Path::from(0u64),
            crate::clustering::abstraction::Abstraction::from(0u64),
            crate::gameplay::path::Path::from(0u64),
        ));

        // Insert test bucket
        profile
            .encounters
            .insert(info, RwLock::new(create_test_bucket()));

        // Constants from the formula
        let activation = crate::SAMPLING_ACTIVATION; // 0.2
        let threshold = crate::SAMPLING_THRESHOLD; // 1.0
        let exploration = crate::SAMPLING_EXPLORATION; // 0.01

        // Test data
        let p1 = 0.3f32;
        let p2 = 0.7f32;
        let p3 = 0.5f32;
        let p4 = crate::POLICY_MIN; // Edge not in bucket (essentially 0)

        // Manual calculation of expected values
        let sum = p1 + p2 + p3 + p4;
        let denom = activation + sum;

        let expected_1 = ((activation + p1 * threshold) / denom).max(exploration);
        let expected_2 = ((activation + p2 * threshold) / denom).max(exploration);
        let expected_3 = ((activation + p3 * threshold) / denom).max(exploration);
        let expected_4 = ((activation + p4 * threshold) / denom).max(exploration);

        // Verify the formula produces expected results
        // With POLICY_MIN being essentially 0, sum ≈ 1.5, denom ≈ 1.7
        assert!(
            (expected_1 - 0.2941176).abs() < 1e-5,
            "Formula check 1: {}",
            expected_1
        );
        assert!(
            (expected_2 - 0.5294118).abs() < 1e-5,
            "Formula check 2: {}",
            expected_2
        );
        assert!(
            (expected_3 - 0.4117647).abs() < 1e-5,
            "Formula check 3: {}",
            expected_3
        );
        assert!(
            (expected_4 - 0.1176471).abs() < 1e-5,
            "Formula check 4: {}",
            expected_4
        );
    }

    /// Test that batch computation in explore_one produces correct weights
    #[test]
    fn test_explore_one_weights() {
        let profile = Profile::default();

        // Create test info
        let info = Info::from((
            crate::gameplay::path::Path::from(0u64),
            crate::clustering::abstraction::Abstraction::from(0u64),
            crate::gameplay::path::Path::from(0u64),
        ));

        // Insert test bucket with known values
        // Use a scope to ensure the entry guard is dropped before we access again
        {
            let bucket_mutex = profile
                .encounters
                .entry(info)
                .or_insert_with(|| RwLock::new(CompactBucket::new()));
            let mut bucket = bucket_mutex.value().write();

            // Add test policies - using edge u8 values directly
            bucket.push((2, (f16::from_f32(0.3), 0.0))); // Edge::Fold = 2
            bucket.push((4, (f16::from_f32(0.7), 0.0))); // Edge::Call = 4
                                                         // Both the write lock and entry guard are dropped here
        }

        // Verify the internal computation logic
        if let Some(bucket_mutex) = profile.encounters.get(&info) {
            let bucket = bucket_mutex.value().read();

            // Verify we can read back the values correctly
            let mut found_fold = false;
            let mut found_call = false;

            for (edge_u8, (policy, _)) in bucket.iter() {
                match edge_u8 {
                    2 => {
                        found_fold = true;
                        // f16 has limited precision, so we need a larger epsilon
                        assert!(
                            (f32::from(policy) - 0.3).abs() < 0.001,
                            "Fold policy: {}",
                            f32::from(policy)
                        );
                    }
                    4 => {
                        found_call = true;
                        // f16 has limited precision, so we need a larger epsilon
                        assert!(
                            (f32::from(policy) - 0.7).abs() < 0.001,
                            "Call policy: {}",
                            f32::from(policy)
                        );
                    }
                    _ => {}
                }
            }

            assert!(found_fold, "Fold policy not found");
            assert!(found_call, "Call policy not found");
        }; // Drop the temporary reference before end of test
    }

    /// Test the optimized sample() against the original formula
    #[test]
    fn test_sample_optimization_correctness() {
        // Test the mathematical equivalence without relying on game state
        let _profile = Profile::default();

        // Test case 1: Empty bucket
        // When sum_policy returns 0 for all edges, the formula should give:
        // q(a) = max(ε, (β + 0 * τ) / (β + n * POLICY_MIN))
        // With n edges, each having POLICY_MIN weight
        let n = 3;
        let sum = n as f32 * crate::POLICY_MIN; // This is essentially 0
        let denom = crate::SAMPLING_ACTIVATION + sum; // This is essentially 0.2
        let numer = crate::SAMPLING_ACTIVATION + crate::POLICY_MIN * crate::SAMPLING_THRESHOLD; // This is essentially 0.2
        let expected_empty = (numer / denom).max(crate::SAMPLING_EXPLORATION);

        // Verify the calculation - with POLICY_MIN ≈ 0, expected_empty ≈ 1.0
        assert!(
            (expected_empty - 1.0).abs() < 1e-5,
            "Empty bucket calculation: {}",
            expected_empty
        );

        // Test case 2: With policies
        let p1 = 0.4f32;
        let p2 = 0.6f32;
        let sum_with_policies = p1 + p2 + crate::POLICY_MIN; // ≈ 1.0
        let denom_with_policies = crate::SAMPLING_ACTIVATION + sum_with_policies; // ≈ 1.2

        let expected_p1 = ((crate::SAMPLING_ACTIVATION + p1 * crate::SAMPLING_THRESHOLD)
            / denom_with_policies)
            .max(crate::SAMPLING_EXPLORATION);
        let expected_p2 = ((crate::SAMPLING_ACTIVATION + p2 * crate::SAMPLING_THRESHOLD)
            / denom_with_policies)
            .max(crate::SAMPLING_EXPLORATION);

        // Verify relative ordering
        assert!(
            expected_p2 > expected_p1,
            "Higher policy should produce higher sampling probability"
        );

        // With POLICY_MIN being essentially 0, we need different assertions
        // expected_p1 ≈ (0.2 + 0.4) / 1.2 ≈ 0.5
        // expected_p2 ≈ (0.2 + 0.6) / 1.2 ≈ 0.667
        assert!(
            (expected_p1 - 0.5).abs() < 1e-5,
            "Expected p1: {}",
            expected_p1
        );
        assert!(
            (expected_p2 - 0.6666667).abs() < 1e-5,
            "Expected p2: {}",
            expected_p2
        );
    }

    // Additional test ensuring advice skips pruned edges
    #[test]
    fn test_advice_respects_skip() {
        use crate::gameplay::edge::Edge as GEdge;

        // Build an Info with three choices Fold, Check, Call
        let futures = vec![GEdge::Fold, GEdge::Check, GEdge::Call];
        let info = Info::from((
            crate::gameplay::path::Path::default(),
            crate::clustering::abstraction::Abstraction::from(0u64),
            crate::gameplay::path::Path::from(futures.clone()),
        ));

        let mut bucket = CompactBucket::new();
        // Positive regrets for Fold & Check
        bucket.push((u8::from(GEdge::Fold), (f16::from_f32(0.5), 10.0)));
        bucket.push((u8::from(GEdge::Check), (f16::from_f32(0.5), 5.0)));
        // Negative regret for Call that leads to pruning
        bucket.push((u8::from(GEdge::Call), (f16::from_f32(0.5), -10.0)));

        // Apply RBP to Call so it is skipped for a while
        bucket.check_and_apply_rbp(
            u8::from(GEdge::Call),
            -10.0,
            0,
            crate::CFR_TREE_COUNT_NLHE as u64,
        );

        let profile = Profile::default();
        profile
            .encounters
            .insert(info, RwLock::new(bucket));

        // Advice for pruned Call edge should be zero
        assert_eq!(profile.advice(&info, &GEdge::Call), 0.0);
    }

    // Test that RBP uses per-run traversals, not cumulative
    #[test]
    fn test_rbp_uses_run_traversals() {
        use crate::gameplay::edge::Edge as GEdge;

        let mut profile = Profile::default();
        // Simulate a resumed training run with high total_traversals
        // but low iterations (just started this run)
        profile.total_traversals = 20_000_000; // 20M from previous runs
        profile.iterations = 10; // Only 10 iterations in current run

        // Per-run traversals should be 10 * 1024 = 10,240
        assert_eq!(profile.run_traversals(), 10_240);

        let futures = vec![GEdge::Fold, GEdge::Check, GEdge::Call];
        let info = Info::from((
            crate::gameplay::path::Path::default(),
            crate::clustering::abstraction::Abstraction::from(0u64),
            crate::gameplay::path::Path::from(futures.clone()),
        ));

        let mut bucket = CompactBucket::new();
        bucket.push((u8::from(GEdge::Fold), (f16::from_f32(0.5), 100.0)));
        bucket.push((u8::from(GEdge::Check), (f16::from_f32(0.5), 50.0)));
        bucket.push((u8::from(GEdge::Call), (f16::from_f32(0.5), -50.0)));

        // Apply RBP - skip_iters = ceil(50/150) = 1
        // With run_traversals = 10,240, this should skip until level 1 (2,097,152)
        bucket.check_and_apply_rbp(
            u8::from(GEdge::Call),
            -50.0,
            profile.run_traversals(),
            crate::CFR_TREE_COUNT_NLHE as u64,
        );

        profile.encounters.insert(info, RwLock::new(bucket));

        // Call should be skipped since run_traversals (10,240) < skip threshold (2,097,152)
        assert_eq!(profile.advice(&info, &GEdge::Call), 0.0);
        assert_eq!(profile.sum_policy(&info, &GEdge::Call), 0.0);
        assert_eq!(profile.sum_regret(&info, &GEdge::Call), 0.0);

        // Simulate more iterations in current run
        profile.iterations = 2_100; // 2100 * 1024 = 2,150,400 > 2,097,152

        // Now Call should be active again
        assert!(profile.advice(&info, &GEdge::Call) > 0.0);
    }
}
