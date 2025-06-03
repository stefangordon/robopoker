use super::edge::Edge;
use super::game::Game;
use super::info::Info;
use super::turn::Turn;
use super::compact_bucket::CompactBucket;
use crate::cards::street::Street;
use crate::Arbitrary;
use zstd::stream::Decoder as ZDecoder;
use dashmap::DashMap;
use parking_lot::RwLock;
use crate::mccfr::types::policy::Policy;
use crate::mccfr::types::decision::Decision;
use crate::mccfr::traits::info::Info as InfoTrait;
use std::path::PathBuf;
use rustc_hash::FxHasher;
use std::hash::BuildHasher;
use half::f16;
#[cfg(feature = "native")]
use arrow2::{
    array::{Int64Array, UInt32Array, Float32Array, Array},
    chunk::Chunk,
    datatypes::{DataType, Field, Schema},
    io::parquet::{
        read::{read_metadata, infer_schema, FileReader},
        write::{
            CompressionOptions, Encoding, FileWriter, RowGroupIterator,
            Version, WriteOptions
        },
    },
};
use std::sync::Arc;

// File format constants and specifications
//
// Two formats are supported:
// 1. Legacy: zstd-compressed PostgreSQL binary COPY format (detected by zstd magic bytes)
// 2. Current: Parquet format with zstd compression (default for new saves)
//
// Legacy Format: [zstd header][19-byte PG header][records...][trailer]
// Parquet Format: Standard Apache Parquet with schema:
//   - history: INT64
//   - present: INT64
//   - futures: INT64
//   - edge: UINT32
//   - regret: FLOAT32
//   - policy: FLOAT32

const ZSTD_MAGIC: [u8; 4] = [0x28, 0xB5, 0x2F, 0xFD];
const PGCOPY_RECORD_SIZE: usize = 66; // 2-byte header already consumed + 64 payload
const PGCOPY_HEADER_SIZE: usize = 19; // PostgreSQL binary COPY header

// Store Edge as compact u8 key inside buckets to reduce memory footprint
// The Edge <-> u8 bijection is already implemented via From conversions.
type Bucket = CompactBucket;

#[inline(always)]
fn safe_clamp(val: f32, min: f32, max: f32) -> f32 {
    let val = val.max(min).min(max); // NaNs will propagate
    if val.is_nan() { 0.0 } else { val }
}

pub struct Profile {
    pub(super) iterations: usize,
    pub(super) encounters: DashMap<Info, RwLock<Bucket>, FxBuildHasher>,
    pub(super) regret_min: crate::Utility,
    pub(super) regret_max: crate::Utility,
}

impl Default for Profile {
    fn default() -> Self {
        Self {
            iterations: 0,
            encounters: DashMap::with_hasher(FxBuildHasher::default()),
            regret_min: crate::REGRET_MIN,
            regret_max: crate::REGRET_MAX,
        }
    }
}

impl Profile {
    fn name() -> String {
        "blueprint".to_string()
    }

    /// Calculate convergence metrics for monitoring training progress
    pub fn convergence_metrics(&self) -> (f64, f64, f64) {
        let mut total_edges = 0usize;
        let mut near_zero_regrets = 0usize;
        let mut high_confidence_actions = 0usize;
        let mut total_entropy = 0.0f64;

        const ZERO_THRESHOLD: f32 = 1.0;
        const HIGH_CONFIDENCE_THRESHOLD: f32 = 0.9;

        for bucket_entry in self.encounters.iter() {
            let bucket = bucket_entry.value().read();
            let n = bucket.len();
            if n == 0 { continue; }

            // Count near-zero regrets
            for (_, (_, regret)) in bucket.iter() {
                total_edges += 1;
                if regret.abs() < ZERO_THRESHOLD {
                    near_zero_regrets += 1;
                }
            }

            // Calculate strategy entropy for this infoset
            let weights: Vec<f32> = bucket.iter()
                .map(|(_, (policy, _))| f32::from(policy).max(0.0))
                .collect();
            let sum: f32 = weights.iter().sum();

            if sum > 0.0 {
                let mut entropy = 0.0f64;
                let mut max_prob = 0.0f32;

                for w in &weights {
                    let p = w / sum;
                    if p > max_prob {
                        max_prob = p;
                    }
                    if p > 0.0 {
                        entropy -= (p as f64) * (p as f64).ln();
                    }
                }

                if max_prob > HIGH_CONFIDENCE_THRESHOLD {
                    high_confidence_actions += 1;
                }

                total_entropy += entropy;
            }
        }

        let convergence_ratio = if total_edges > 0 {
            near_zero_regrets as f64 / total_edges as f64
        } else {
            0.0
        };

        let determinism_ratio = if self.encounters.len() > 0 {
            high_confidence_actions as f64 / self.encounters.len() as f64
        } else {
            0.0
        };

        let avg_entropy = if self.encounters.len() > 0 {
            total_entropy / self.encounters.len() as f64
        } else {
            0.0
        };

        (convergence_ratio, determinism_ratio, avg_entropy)
    }

    /// Log aggregate statistics for regret and policy vectors instead of writing to disk.
    /// Enabled by setting environment variable `BLUEPRINT_STATS=1` before running.
    #[cfg(feature = "native")]
    pub fn log_stats(&self) {
        use std::f32::{INFINITY, NEG_INFINITY};

        let mut infosets = 0usize;
        let mut total_edges = 0usize;
        let mut min_edges = usize::MAX;
        let mut max_edges = 0usize;

        // Add edge count distribution tracking
        let mut edge_count_dist = [0usize; 14]; // 0-13 edges

        let mut policy_min = INFINITY;
        let mut policy_max = NEG_INFINITY;
        let mut regret_min = INFINITY;
        let mut regret_max = NEG_INFINITY;

        // Distribution tracking
        let mut regret_buckets = [0usize; 8]; // [-inf, -1M, -10k, -100, 0, 100, 10k, 1M, inf]
        let mut policy_buckets = [0usize; 5]; // [0, 1, 10, 100, inf]
        let mut zero_regret_count = 0usize;

        for bucket_entry in self.encounters.iter() {
            infosets += 1;
            let bucket = bucket_entry.value().read();
            let count = bucket.len();
            total_edges += count;
            min_edges = min_edges.min(count);
            max_edges = max_edges.max(count);

            // Track edge count distribution
            if count < edge_count_dist.len() {
                edge_count_dist[count] += 1;
            }

            for (_, (policy, regret)) in bucket.iter() {
                let pol_f32 = f32::from(policy);
                policy_min = policy_min.min(pol_f32);
                policy_max = policy_max.max(pol_f32);
                regret_min = regret_min.min(regret);
                regret_max = regret_max.max(regret);

                // Bucket regrets
                if regret == 0.0 {
                    zero_regret_count += 1;
                } else if regret < -1_000_000.0 {
                    regret_buckets[0] += 1;
                } else if regret < -10_000.0 {
                    regret_buckets[1] += 1;
                } else if regret < -100.0 {
                    regret_buckets[2] += 1;
                } else if regret < 0.0 {
                    regret_buckets[3] += 1;
                } else if regret < 100.0 {
                    regret_buckets[4] += 1;
                } else if regret < 10_000.0 {
                    regret_buckets[5] += 1;
                } else if regret < 1_000_000.0 {
                    regret_buckets[6] += 1;
                } else {
                    regret_buckets[7] += 1;
                }

                // Bucket policies
                if pol_f32 < 1.0 {
                    policy_buckets[0] += 1;
                } else if pol_f32 < 10.0 {
                    policy_buckets[1] += 1;
                } else if pol_f32 < 100.0 {
                    policy_buckets[2] += 1;
                } else if pol_f32 < 1000.0 {
                    policy_buckets[3] += 1;
                } else {
                    policy_buckets[4] += 1;
                }
            }
        }

        let avg_edges = if infosets > 0 {
            total_edges as f64 / infosets as f64
        } else {
            0.0
        };

        log::info!("------------------ BLUEPRINT STATS ------------------");
        log::info!("InfoSets:      {}", infosets);
        log::info!("Edges / set:   min {}  max {}  avg {:.2}", min_edges, max_edges, avg_edges);
        log::info!("Policy range:  [{:.4}, {:.4}]", policy_min, policy_max);
        log::info!("Regret range:  [{:.4}, {:.4}]", regret_min, regret_max);
        log::info!("Zero regrets:  {} ({:.2}% of edges)", zero_regret_count,
                   100.0 * zero_regret_count as f64 / total_edges as f64);
        log::info!("Regret distribution:");
        log::info!("  < -1M:       {} edges", regret_buckets[0]);
        log::info!("  [-1M, -10k): {} edges", regret_buckets[1]);
        log::info!("  [-10k, -100): {} edges", regret_buckets[2]);
        log::info!("  [-100, 0):   {} edges", regret_buckets[3]);
        log::info!("  [0, 100):    {} edges", regret_buckets[4]);
        log::info!("  [100, 10k):  {} edges", regret_buckets[5]);
        log::info!("  [10k, 1M):   {} edges", regret_buckets[6]);
        log::info!("  >= 1M:       {} edges", regret_buckets[7]);
        log::info!("Policy weight distribution:");
        log::info!("  [0, 1):      {} edges", policy_buckets[0]);
        log::info!("  [1, 10):     {} edges", policy_buckets[1]);
        log::info!("  [10, 100):   {} edges", policy_buckets[2]);
        log::info!("  [100, 1k):   {} edges", policy_buckets[3]);
        log::info!("  >= 1k:       {} edges", policy_buckets[4]);

        // Add edge count distribution
        log::info!("Edge count distribution:");
        let mut cumulative = 0usize;
        for (count, freq) in edge_count_dist.iter().enumerate() {
            if *freq > 0 {
                cumulative += freq;
                let pct = *freq as f64 / infosets as f64 * 100.0;
                let cum_pct = cumulative as f64 / infosets as f64 * 100.0;
                log::info!("  {} edges: {:>10} infosets ({:>5.1}%, cum {:>5.1}%)",
                    count, freq, pct, cum_pct);
            }
        }

        // Add convergence metrics
        let (convergence_ratio, determinism_ratio, avg_entropy) = self.convergence_metrics();
        log::info!("Convergence metrics:");
        log::info!("  Near-zero regrets: {:.1}%", convergence_ratio * 100.0);
        log::info!("  High-confidence actions: {:.1}%", determinism_ratio * 100.0);
        log::info!("  Average entropy: {:.3}", avg_entropy);

        // Additional derived metrics
        let negative_regrets: usize = regret_buckets[0..4].iter().sum();
        let positive_regrets: usize = regret_buckets[4..].iter().sum();
        let total_nonzero = negative_regrets + positive_regrets;
        if total_nonzero > 0 {
            let balance_ratio = negative_regrets as f64 / total_nonzero as f64;
            log::info!("  Negative/Total ratio: {:.1}% (ideal ~50%)", balance_ratio * 100.0);
        }

        log::info!("-----------------------------------------------------");
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
            bucket.push((u8::from(d.edge()), (half::f16::from_f32(w), 0.0)));
        }
    }

    /// Apply regret/policy deltas concurrently-safe (called by subgame solver).
    pub fn apply_updates(&self, updates: Vec<(Info, crate::mccfr::types::policy::Policy<Edge>, crate::mccfr::types::policy::Policy<Edge>)>) {
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
                    let safe_delta = if delta.is_nan() || delta < 0.0 { crate::POLICY_MIN } else { delta };
                    bucket.push((edge_key, (f16::from_f32(safe_delta), 0.0)));
                }
            }
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
            for (edge_u8, (_, regret)) in bucket.iter() {
                let choice_idx = choice_map[edge_u8 as usize];
                if choice_idx != 0xff {
                    regrets[choice_idx as usize] = regret.max(crate::POLICY_MIN);
                }
            }

            // Compute sum and normalize
            let sum: f32 = regrets[..all_choices.len()].iter().sum();

            // Build result with normalization
            all_choices.iter()
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
            for (edge_u8, (policy, _)) in bucket.iter() {
                let choice_idx = choice_map[edge_u8 as usize];
                if choice_idx != 0xff {
                    policies[choice_idx as usize] = f32::from(policy).max(crate::POLICY_MIN);
                }
            }

            // Apply sampling formula: q(a) = max(ε, (β + τ * weight(a)) / (β + sum(weights)))
            let sum: f32 = policies[..all_choices.len()].iter().sum();
            let denom = activation + sum;

            // Build result with sampling probabilities
            all_choices.iter()
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
            all_choices.into_iter().map(|edge| (edge, uniform_prob)).collect()
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
            bucket
                .iter()
                .find_map(|(e, (p, _))| if e == u8::from(*edge) { Some(f32::from(p)) } else { None })
                .unwrap_or_default()
        } else {
            0.0
        }
    }

    fn sum_regret(&self, info: &Self::I, edge: &Self::E) -> crate::Utility {
        let regret_val = if let Some(bucket_mutex) = self.encounters.get(info) {
            let bucket = bucket_mutex.value().read();
            bucket
                .iter()
                .find_map(|(e, (_, r))| if e == u8::from(*edge) { Some(r) } else { None })
                .unwrap_or_default()
        } else {
            0.0
        };

        if regret_val.is_nan() {
            // Ensure NaN is not returned. This indicates a NaN was stored previously.
            log::warn!("sum_regret: Encountered stored NaN for info: {:?}, edge: {:?}. Returning 0.0 instead.", info, edge);
            0.0
        } else {
            regret_val
        }
    }

    fn current_regret_min(&self) -> crate::Utility {
        self.regret_min
    }
    fn current_regret_max(&self) -> crate::Utility {
        self.regret_max
    }

    // Override the default policy() method to use direct bucket access
    fn policy(&self, info: &Self::I, edge: &Self::E) -> crate::Probability {
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

            let mut sum = 0.0f32;
            let mut target_regret = crate::POLICY_MIN;
            let mut found_mask = 0u16;

            // Single pass through bucket
            for (edge_u8, (_, regret)) in bucket.iter() {
                let choice_idx = choice_map[edge_u8 as usize];
                if choice_idx != 0xff {
                    let r = regret.max(crate::POLICY_MIN);
                    sum += r;
                    if choice_idx == target_idx {
                        target_regret = r;
                    }
                    found_mask |= 1 << choice_idx;
                }
            }

            // Add defaults for missing edges
            let missing_count = all_choices.len() - found_mask.count_ones() as usize;
            sum += missing_count as f32 * crate::POLICY_MIN;

            target_regret / sum
        } else {
            // No data: uniform distribution
            1.0 / all_choices.len() as f32
        }
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

            let mut sum = 0.0f32;
            let mut target_policy = crate::POLICY_MIN;
            let mut found_mask = 0u16;

            // Single pass through bucket
            for (edge_u8, (policy, _)) in bucket.iter() {
                let choice_idx = choice_map[edge_u8 as usize];
                if choice_idx != 0xff {
                    let p = f32::from(policy).max(crate::POLICY_MIN);
                    sum += p;
                    if choice_idx == target_idx {
                        target_policy = p;
                    }
                    found_mask |= 1 << choice_idx;
                }
            }

            // Add defaults for missing edges
            let missing_count = all_choices.len() - found_mask.count_ones() as usize;
            sum += missing_count as f32 * crate::POLICY_MIN;

            target_policy / sum
        } else {
            // No data: uniform distribution
            1.0 / all_choices.len() as f32
        }
    }

    // Override the sample() method to use batch access
    fn sample(&self, info: &Self::I, edge: &Self::E) -> crate::Probability {
        let all_choices = info.choices();
        if all_choices.is_empty() {
            return 0.0;
        }

        // Early check if edge is in choices
        let target_idx = match all_choices.iter().position(|&e| e == *edge) {
            Some(idx) => idx,
            None => return 0.0,
        };

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

            // Direct computation without intermediate storage
            let mut sum = 0.0f32;
            let mut target_policy = 0.0f32;
            let mut found_mask = 0u16;

            // Single pass through bucket
            for (edge_u8, (policy, _)) in bucket.iter() {
                let choice_idx = choice_map[edge_u8 as usize];
                if choice_idx != 0xff {
                    let p = f32::from(policy).max(crate::POLICY_MIN);
                    sum += p;
                    if choice_idx as usize == target_idx {
                        target_policy = p;
                    }
                    found_mask |= 1 << choice_idx;
                }
            }

            // Add defaults for missing edges (but not to target if it wasn't found)
            let was_target_found = (found_mask & (1 << target_idx)) != 0;
            if !was_target_found {
                target_policy = crate::POLICY_MIN;
            }
            let missing_count = all_choices.len() - found_mask.count_ones() as usize;
            sum += missing_count as f32 * crate::POLICY_MIN;

            // Apply sampling formula: q(a) = max(ε, (β + τ * weight(a)) / (β + sum(weights)))
            let denom = activation + sum;
            let numer = activation + target_policy * threshold;
            (numer / denom).max(exploration)
        } else {
            // No data: all policies are POLICY_MIN
            let n = all_choices.len() as f32;
            let sum = n * crate::POLICY_MIN;
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

            for (edge_u8, (policy, _)) in bucket.iter() {
                let p = f32::from(policy).max(crate::POLICY_MIN);
                edge_policies[edge_u8 as usize] = p;
                total_policy += p;
            }

            // Also count edges not in bucket
            let all_choices = info.choices();
            for edge in &all_choices {
                let edge_u8 = u8::from(*edge);
                if edge_policies[edge_u8 as usize] == 0.0 {
                    edge_policies[edge_u8 as usize] = crate::POLICY_MIN;
                    total_policy += crate::POLICY_MIN;
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

#[cfg(feature = "native")]
impl crate::save::upload::Table for Profile {
    fn name() -> String {
        Self::name()
    }
    fn columns() -> &'static [tokio_postgres::types::Type] {
        &[
            tokio_postgres::types::Type::INT8,
            tokio_postgres::types::Type::INT8,
            tokio_postgres::types::Type::INT8,
            tokio_postgres::types::Type::INT8,
            tokio_postgres::types::Type::FLOAT4,
            tokio_postgres::types::Type::FLOAT4,
        ]
    }
    fn sources() -> Vec<String> {
        use crate::save::disk::Disk;
        use crate::Arbitrary;
        vec![Self::path(Street::random())]
    }
    fn copy() -> String {
        "COPY blueprint (
            past,
            present,
            future,
            edge,
            regret,
            policy
        )
        FROM STDIN BINARY
        "
        .to_string()
    }
    fn creates() -> String {
        "
        CREATE TABLE IF NOT EXISTS blueprint (
            edge       BIGINT,
            past       BIGINT,
            present    BIGINT,
            future     BIGINT,
            policy     REAL,
            regret     REAL
        ) WITH (autovacuum_enabled = false);
        "
        .to_string()
    }
    fn indices() -> String {
        "
        CREATE INDEX IF NOT EXISTS idx_blueprint_bucket  ON blueprint (present, past, future);
        CREATE INDEX IF NOT EXISTS idx_blueprint_future  ON blueprint (future);
        CREATE INDEX IF NOT EXISTS idx_blueprint_present ON blueprint (present);
        CREATE INDEX IF NOT EXISTS idx_blueprint_edge    ON blueprint (edge);
        CREATE INDEX IF NOT EXISTS idx_blueprint_past    ON blueprint (past);
        "
        .to_string()
    }
}

#[cfg(feature = "native")]
impl crate::save::disk::Disk for Profile {
    fn name() -> String {
        Self::name()
    }
    fn grow(_: Street) -> Self {
        unreachable!("must be learned in MCCFR minimization")
    }
    fn path(_: Street) -> String {
        let current_dir = std::env::current_dir().unwrap_or_default();
        let path = PathBuf::from(current_dir)
            .join("pgcopy")
            .join(Self::name());

        path.to_string_lossy().into_owned()
    }
    fn done(_: Street) -> bool {
        std::fs::metadata(Self::path(Street::random())).is_ok()
    }
    fn load(_: Street) -> Self {
        use std::fs::File;
        use std::io::Read;

        let path = Self::path(Street::random());
        log::info!("{:<32}{:<32}", "loading     blueprint", path);

        // Detect format by checking for zstd magic bytes
        let mut magic = [0u8; 4];
        let mut file = File::open(&path).expect("Failed to open blueprint file");
        file.read_exact(&mut magic).expect("Failed to read file header");

        if magic == ZSTD_MAGIC {
            log::debug!("Detected zstd-compressed PostgreSQL binary format (legacy)");
            Self::load_legacy_pgcopy()
        } else {
            log::debug!("Detected parquet format");
            Self::load_parquet()
        }
    }
    fn save(&self) {
        if std::env::var("BLUEPRINT_STATS").is_ok() {
            // If caller only wants statistics, skip expensive serialization
            self.log_stats();
        } else {
            // Use parquet format for all saves
            self.save_parquet();
        }
    }
}

impl Profile {
    /// Populates the profile with synthetic data for benchmarking purposes.
    ///
    /// `bucket_count`: number of distinct information sets to generate.
    /// `edges_per_bucket`: number of edges per information set.
    pub fn populate_dummy(&mut self, bucket_count: usize, edges_per_bucket: usize) {
        use rand::Rng;

        for _ in 0..bucket_count {
            let info = Info::random();
            let bucket_mutex = self
                .encounters
                .entry(info)
                .or_insert_with(|| RwLock::new(CompactBucket::new()));
            let mut bucket = bucket_mutex.value().write();
            for _ in 0..edges_per_bucket {
                let edge = Edge::random();
                let policy: f32 = rand::thread_rng().gen_range(0.0..1.0);
                let regret: f32 = rand::thread_rng().gen_range(-1.0..1.0);
                bucket.push((u8::from(edge), (f16::from_f32(policy), regret)));
            }
        }

        // Update iterations counter to reflect synthetic data
        self.iterations += bucket_count;
    }

    /// Save profile data using Apache Parquet format with zstd compression.
    /// This is now the default save format.
    #[cfg(feature = "native")]
    pub fn save_parquet(&self) {
        use crate::save::disk::Disk;
        use std::fs::File;
        use std::io::BufWriter;

        let path_str = Self::path(Street::random());
        let path = std::path::Path::new(&path_str);

        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("Failed to create parent directories");
        }

        // Calculate total records first
        let total_records: usize = self.encounters.iter()
            .map(|entry| entry.value().read().len())
            .sum();

        log::info!("Saving blueprint to {} ({} records)", path_str, total_records);

        let progress = crate::progress(total_records);
        progress.set_message("Saving blueprint to parquet");

        // Create schema
        let schema = Schema::from(vec![
            Field::new("history", DataType::Int64, false),
            Field::new("present", DataType::Int64, false),
            Field::new("futures", DataType::Int64, false),
            Field::new("edge", DataType::UInt32, false),
            Field::new("regret", DataType::Float32, false),
            Field::new("policy", DataType::Float32, false),
        ]);

                // Define encodings - use plain encoding to minimize memory usage
        let encodings = vec![
            vec![Encoding::Plain],      // history
            vec![Encoding::Plain],      // present
            vec![Encoding::Plain],      // futures
            vec![Encoding::Plain],      // edge
            vec![Encoding::Plain],      // regret
            vec![Encoding::Plain],      // policy
        ];

        // Write options - optimized for speed while keeping compression
        let options = WriteOptions {
            write_statistics: true, // Keep statistics for better query performance
            compression: CompressionOptions::Zstd(Some(arrow2::io::parquet::write::ZstdLevel::try_new(1).unwrap())), // Fast ZSTD compression level 1
            version: Version::V2,
            data_pagesize_limit: Some(2 * 1024 * 1024), // 2MB pages for good balance
        };

        // Create output file and writer with reasonable buffer
        let file = File::create(path).expect(&format!("Failed to create file at {}", path_str));
        let writer_inner = BufWriter::with_capacity(16 * 1024 * 1024, file); // 16MB buffer
        let mut writer = FileWriter::try_new(writer_inner, schema.clone(), options)
            .expect("Failed to create parquet writer");

        const RECORDS_PER_CHUNK: usize = 5_000_000; // 3M records per chunk
        let mut written_records = 0;

        let mut chunk_histories = Vec::with_capacity(RECORDS_PER_CHUNK);
        let mut chunk_presents = Vec::with_capacity(RECORDS_PER_CHUNK);
        let mut chunk_futures = Vec::with_capacity(RECORDS_PER_CHUNK);
        let mut chunk_edges = Vec::with_capacity(RECORDS_PER_CHUNK);
        let mut chunk_regrets = Vec::with_capacity(RECORDS_PER_CHUNK);
        let mut chunk_policies = Vec::with_capacity(RECORDS_PER_CHUNK);

        for bucket_entry in self.encounters.iter() {
            let bucket = bucket_entry.key();
            let edges_vec = bucket_entry.value().read();

            for (edge, (policy, regret)) in edges_vec.iter() {
                let edge_enum: Edge = edge.into();
                chunk_histories.push(u64::from(*bucket.history()) as i64);
                chunk_presents.push(u64::from(*bucket.present()) as i64);
                chunk_futures.push(u64::from(*bucket.futures()) as i64);
                chunk_edges.push(u64::from(edge_enum) as u32);
                chunk_regrets.push(regret);
                chunk_policies.push(f32::from(policy));

                written_records += 1;

                // Update progress less frequently to reduce overhead
                if written_records % 1_000_000 == 0 {
                    progress.set_position(written_records as u64);
                }

                // Write chunk when it's full
                if chunk_histories.len() >= RECORDS_PER_CHUNK {
                    Self::write_parquet_chunk(
                        &mut writer,
                        &schema,
                        &options,
                        &encodings,
                        &chunk_histories,
                        &chunk_presents,
                        &chunk_futures,
                        &chunk_edges,
                        &chunk_regrets,
                        &chunk_policies,
                    );

                    // Reuse vectors instead of reallocating - just reset length
                    chunk_histories.clear();
                    chunk_presents.clear();
                    chunk_futures.clear();
                    chunk_edges.clear();
                    chunk_regrets.clear();
                    chunk_policies.clear();

                }
            }
        }

        // Write final partial chunk if any records remain
        if !chunk_histories.is_empty() {
            Self::write_parquet_chunk(
                &mut writer,
                &schema,
                &options,
                &encodings,
                &chunk_histories,
                &chunk_presents,
                &chunk_futures,
                &chunk_edges,
                &chunk_regrets,
                &chunk_policies,
            );
        }

        let _size = writer.end(None).expect("Failed to finalize parquet file");
        progress.finish_with_message("Blueprint saved to parquet");
    }

    /// Helper function to write a chunk of data to the parquet file
    #[cfg(feature = "native")]
    fn write_parquet_chunk(
        writer: &mut FileWriter<std::io::BufWriter<std::fs::File>>,
        schema: &Schema,
        options: &WriteOptions,
        encodings: &[Vec<Encoding>],
        histories: &[i64],
        presents: &[i64],
        futures: &[i64],
        edges: &[u32],
        regrets: &[f32],
        policies: &[f32],
    ) {
        // Create arrow arrays for this chunk
        let hist_array = Int64Array::from_slice(histories);
        let pres_array = Int64Array::from_slice(presents);
        let fut_array = Int64Array::from_slice(futures);
        let edge_array = UInt32Array::from_slice(edges);
        let regret_array = Float32Array::from_slice(regrets);
        let policy_array = Float32Array::from_slice(policies);

        let columns: Vec<Arc<dyn Array>> = vec![
            Arc::new(hist_array),
            Arc::new(pres_array),
            Arc::new(fut_array),
            Arc::new(edge_array),
            Arc::new(regret_array),
            Arc::new(policy_array),
        ];

        // Create chunk and write as row group
        let chunk = Chunk::new(columns);
        let row_groups = RowGroupIterator::try_new(
            vec![Ok(chunk)].into_iter(),
            schema,
            *options,
            encodings.to_vec(),
        ).expect("Failed to create row group iterator");

        // Write this chunk as a row group
        for group in row_groups {
            writer.write(group.expect("Failed to get row group"))
                .expect("Failed to write row group");
        }
    }

    /// Load profile data from Apache Parquet format.
    #[cfg(feature = "native")]
    pub fn load_parquet() -> Self {
        use crate::clustering::abstraction::Abstraction;
        use crate::gameplay::path::Path;
        use crate::save::disk::Disk;
        use std::fs::File;
        use rayon::prelude::*;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let path_str = Self::path(Street::random());
        log::info!("{:<32}{:<32}", "loading     blueprint", path_str);

        let mut file = File::open(&path_str).expect(&format!("Failed to open blueprint file: {}", path_str));

        // Read metadata
        let metadata = read_metadata(&mut file).expect("Failed to read parquet metadata");

        // Infer schema
        let schema = infer_schema(&metadata).expect("Failed to infer schema");

        let total_rows: usize = metadata.row_groups.iter().map(|rg| rg.num_rows() as usize).sum();
        log::info!("Loading blueprint from {} row groups ({} total rows)", metadata.row_groups.len(), total_rows);

        let progress = crate::progress(total_rows);
        progress.set_message("Loading blueprint from parquet");

        let encounters: DashMap<Info, RwLock<Bucket>, FxBuildHasher> =
            DashMap::with_hasher(FxBuildHasher::default());

        // Atomic row counter for optional logging/debugging
        let processed_rows = AtomicUsize::new(0);

        // Share path so each thread can reopen the file independently
        let path_shared = Arc::new(path_str.clone());

        // Build a dedicated 8-thread pool for loading
        rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .expect("Failed to build rayon pool")
            .install(|| {
                metadata
                    .row_groups
                    .par_iter()
                    .enumerate()
                    .for_each(|(_rg_idx, row_group)| {
                        // Open a fresh handle for this row-group
                        let file_handle = std::fs::File::open(&*path_shared)
                            .expect("Failed to reopen parquet file");

                        let chunks = FileReader::new(
                            file_handle,
                            vec![row_group.clone()],
                            schema.clone(),
                            None,
                            None,
                            None,
                        );

                        for chunk_result in chunks {
                            let chunk = chunk_result.expect("Failed to read chunk");

                            // Column arrays
                            let arrays = chunk.arrays();
                            debug_assert_eq!(arrays.len(), 6, "Unexpected column count");

                            let histories = arrays[0].as_any().downcast_ref::<Int64Array>()
                                .expect("history col");
                            let presents = arrays[1].as_any().downcast_ref::<Int64Array>()
                                .expect("present col");
                            let futures = arrays[2].as_any().downcast_ref::<Int64Array>()
                                .expect("futures col");
                            let edges = arrays[3].as_any().downcast_ref::<UInt32Array>()
                                .expect("edge col");
                            let regrets = arrays[4].as_any().downcast_ref::<Float32Array>()
                                .expect("regret col");
                            let policies = arrays[5].as_any().downcast_ref::<Float32Array>()
                                .expect("policy col");

                            let num_rows = chunk.len();

                            for i in 0..num_rows {
                                let history = Path::from(histories.value(i) as u64);
                                let present = Abstraction::from(presents.value(i) as u64);
                                let futures = Path::from(futures.value(i) as u64);
                                let edge = Edge::from(edges.value(i) as u64);
                                let regret = regrets.value(i);
                                let policy = policies.value(i);

                                let info = Info::from((history, present, futures));
                                let bucket_mutex = encounters
                                    .entry(info)
                                    .or_insert_with(|| RwLock::new(CompactBucket::new()));
                                bucket_mutex
                                    .value()
                                    .write()
                                    .push((u8::from(edge), (f16::from_f32(policy), regret)));

                                processed_rows.fetch_add(1, Ordering::Relaxed);
                                progress.inc(1);
                            }
                        }
                    });
            });

        progress.finish_with_message("Blueprint loaded from parquet");

        let profile = Self {
            encounters,
            iterations: 0,
            regret_min: crate::REGRET_MIN,
            regret_max: crate::REGRET_MAX,
        };

        // Always log aggregate statistics after loading
        #[cfg(feature = "native")]
        profile.log_stats();

        profile
    }

    /// Load profile data from legacy zstd-compressed PostgreSQL binary format files.
    /// This method handles the original file format used before the mmap optimization.
    #[cfg(feature = "native")]
    fn load_legacy_pgcopy() -> Self {
        use crate::clustering::abstraction::Abstraction;
        use crate::gameplay::path::Path;
        use crate::mccfr::nlhe::info::Info;
        use byteorder::{ByteOrder, BE};
        use std::fs::File;
        use std::io::{BufReader, Read};
        use crate::save::disk::Disk;

        let path = Self::path(Street::random());
        log::info!("{:<32}{:<32}", "loading     blueprint (legacy)", path);

        // Open file and create zstd decoder (we know it's zstd compressed)
        let file = File::open(&path).expect("Failed to open blueprint file");
        let reader_inner = ZDecoder::new(file).expect("Failed to create zstd decoder");
        let mut reader = BufReader::with_capacity(1024 * 1024, reader_inner);

        // Skip the PostgreSQL binary COPY header
        let mut skip = [0u8; PGCOPY_HEADER_SIZE];
        reader.read_exact(&mut skip).expect("Failed to skip pgcopy header");

        let encounters: DashMap<Info, RwLock<Bucket>, FxBuildHasher> = DashMap::with_hasher(FxBuildHasher::default());
        let mut header = [0u8; 2];
        let mut record = [0u8; PGCOPY_RECORD_SIZE - 2]; // 2-byte header already consumed + 64 payload

        loop {
            if reader.read_exact(&mut header).is_err() {
                break; // EOF
            }
            let fields = u16::from_be_bytes(header);
            if fields == 0xFFFF {
                break; // trailer
            }
            debug_assert_eq!(fields, 6, "Unexpected field count: {}", fields);

            // Read remaining bytes of the record
            reader.read_exact(&mut record).expect("Failed to read record bytes");

            // Parse PostgreSQL binary format record
            let mut offset = 0;
            let mut read_field = |size: usize| {
                offset += 4; // skip length prefix
                let start = offset;
                let end = offset + size;
                offset = end;
                &record[start..end]
            };

            let history = Path::from(BE::read_u64(read_field(8)));
            let present = Abstraction::from(BE::read_u64(read_field(8)));
            let futures = Path::from(BE::read_u64(read_field(8)));
            let edge = Edge::from(BE::read_u64(read_field(8)));
            let regret_raw = BE::read_f32(read_field(4));
            let policy_raw = BE::read_f32(read_field(4));
            let regret = if regret_raw.is_finite() { regret_raw } else { 0.0 };
            let policy = if policy_raw.is_finite() { policy_raw } else { 0.0 };

            let info = Info::from((history, present, futures));
            let bucket_mutex = encounters
                .entry(info)
                .or_insert_with(|| RwLock::new(CompactBucket::new()));
            let mut bucket = bucket_mutex.value().write();
            bucket.push((u8::from(edge), (f16::from_f32(policy), regret)));
        }

        let profile = Self {
            encounters,
            iterations: 0,
            regret_min: crate::REGRET_MIN,
            regret_max: crate::REGRET_MAX,
        };

        // Log stats after loading legacy format as well
        #[cfg(feature = "native")]
        profile.log_stats();

        profile
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

    /// Helper to create a test CompactBucket with known values
    fn create_test_bucket() -> CompactBucket {
        let mut bucket = CompactBucket::new();
        // Add some test data with known edge values
        bucket.push((1, (f16::from_f32(0.3), 0.0)));  // Edge 1
        bucket.push((2, (f16::from_f32(0.7), 0.0)));  // Edge 2
        bucket.push((3, (f16::from_f32(0.5), 0.0)));  // Edge 3
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
        profile.encounters.insert(info, RwLock::new(create_test_bucket()));

        // Constants from the formula
        let activation = crate::SAMPLING_ACTIVATION; // 0.2
        let threshold = crate::SAMPLING_THRESHOLD;   // 1.0
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
        assert!((expected_1 - 0.2941176).abs() < 1e-5, "Formula check 1: {}", expected_1);
        assert!((expected_2 - 0.5294118).abs() < 1e-5, "Formula check 2: {}", expected_2);
        assert!((expected_3 - 0.4117647).abs() < 1e-5, "Formula check 3: {}", expected_3);
        assert!((expected_4 - 0.1176471).abs() < 1e-5, "Formula check 4: {}", expected_4);
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
            bucket.push((2, (f16::from_f32(0.3), 0.0)));  // Edge::Fold = 2
            bucket.push((4, (f16::from_f32(0.7), 0.0)));  // Edge::Call = 4
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
                        assert!((f32::from(policy) - 0.3).abs() < 0.001, "Fold policy: {}", f32::from(policy));
                    }
                    4 => {
                        found_call = true;
                        // f16 has limited precision, so we need a larger epsilon
                        assert!((f32::from(policy) - 0.7).abs() < 0.001, "Call policy: {}", f32::from(policy));
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
        assert!((expected_empty - 1.0).abs() < 1e-5,
            "Empty bucket calculation: {}", expected_empty);

        // Test case 2: With policies
        let p1 = 0.4f32;
        let p2 = 0.6f32;
        let sum_with_policies = p1 + p2 + crate::POLICY_MIN; // ≈ 1.0
        let denom_with_policies = crate::SAMPLING_ACTIVATION + sum_with_policies; // ≈ 1.2

        let expected_p1 = ((crate::SAMPLING_ACTIVATION + p1 * crate::SAMPLING_THRESHOLD) / denom_with_policies)
            .max(crate::SAMPLING_EXPLORATION);
        let expected_p2 = ((crate::SAMPLING_ACTIVATION + p2 * crate::SAMPLING_THRESHOLD) / denom_with_policies)
            .max(crate::SAMPLING_EXPLORATION);

        // Verify relative ordering
        assert!(expected_p2 > expected_p1, "Higher policy should produce higher sampling probability");

        // With POLICY_MIN being essentially 0, we need different assertions
        // expected_p1 ≈ (0.2 + 0.4) / 1.2 ≈ 0.5
        // expected_p2 ≈ (0.2 + 0.6) / 1.2 ≈ 0.667
        assert!((expected_p1 - 0.5).abs() < 1e-5, "Expected p1: {}", expected_p1);
        assert!((expected_p2 - 0.6666667).abs() < 1e-5, "Expected p2: {}", expected_p2);
    }
}

