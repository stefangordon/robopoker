use half::f16;
use std::fmt;

// Stable-RBP constants
const EDGE_MASK: u8 = 0x0F;      // Bottom 4 bits for edge value
const SKIP_MASK: u8 = 0xF0;      // Top 4 bits for skip level
const SKIP_LEVEL_ACTIVE: u8 = 0; // Not skipped
const SKIP_LEVEL_TOMBSTONE: u8 = 15; // Permanently pruned

// Skip iteration lookup table – evenly spaced up to ~252 million traversals.
// Horizon (level 15) now matches 0xF000000 = 251,658,240 iterations.
// Each step is 0x1000000 = 16,777,216 traversals (~16 M).
// If you change `CFR_TREE_COUNT_NLHE` via TOML, adjust this table to keep
// level 15 ≥ horizon, otherwise edges may never tombstone.
const SKIP_ITERATIONS: [u32; 16] = [
    0x0000000,  // 0  – Active
    0x1000000,  // 1  – 16.7 M
    0x2000000,  // 2  – 33.5 M
    0x3000000,  // 3  – 50.3 M
    0x4000000,  // 4  – 67.1 M
    0x5000000,  // 5  – 83.9 M
    0x6000000,  // 6  – 100.7 M
    0x7000000,  // 7  – 117.6 M
    0x8000000,  // 8  – 134.4 M
    0x9000000,  // 9  – 151.2 M
    0xA000000,  // 10 – 168.0 M
    0xB000000,  // 11 – 184.8 M
    0xC000000,  // 12 – 201.7 M
    0xD000000,  // 13 – 218.5 M
    0xE000000,  // 14 – 235.3 M
    0xF000000,  // 15 – 251.6 M (tombstone threshold)
];

/// Find the skip level for a given target iteration
#[inline(always)]
fn encode_skip_until_iter(current_iter: u64, skip_iters: u64) -> u8 {
    let target_iter = current_iter + skip_iters;

    // If target doesn't even reach the first threshold, don't bother encoding
    // (skip_iters is too small to be meaningful)
    if target_iter < SKIP_ITERATIONS[1] as u64 {
        return SKIP_LEVEL_ACTIVE;
    }

    // If target is beyond horizon, tombstone
    if target_iter >= SKIP_ITERATIONS[15] as u64 {
        return SKIP_LEVEL_TOMBSTONE;
    }

    // Find the largest level whose threshold is <= target_iter
    // This gives us "the nearest skip iter that is not larger"
    for level in (1..15).rev() {
        if SKIP_ITERATIONS[level] as u64 <= target_iter {
            return level as u8;
        }
    }

    SKIP_LEVEL_ACTIVE
}

/// A specialized container optimized for poker edge storage.
/// Similar to SmallVec but with better memory layout for our specific use case.
///
/// Stores up to 4 edges inline, spilling to heap for larger collections.
/// Memory layout is optimized for the common case (2-4 edges, 81% cumulative).
///
/// # Stable-RBP Integration
///
/// This implementation supports Regret-Based Pruning (RBP) with zero memory overhead:
/// - The top 4 bits of each edge byte store skip information
/// - Skip levels 0-15 map to iteration milestones evenly spaced across training
/// - Level 0: Active (not skipped)
/// - Levels 1-14: Skip until reaching 2.1M, 4.2M, ..., 29.4M iterations
/// - Level 15: Tombstoned (permanent pruning at 33.5M iterations)
///
/// ## How it works:
/// 1. During regret updates, if new regret is negative, check_and_apply_rbp() is called
/// 2. It calculates skip_iters = ceil(-regret / sum_positive_regrets)
/// 3. Maps current_iter + skip_iters to the next skip level threshold
/// 4. Regular iter() skips tombstoned edges; iter_active() also skips temporarily pruned edges
/// 5. Edges automatically "wake up" when total_traversals passes their skip threshold
pub struct CompactBucket {
    inner: CompactBucketInner,
}

#[repr(u8)]
enum CompactBucketInner {
    Small {
        count: u8,
        edges: [u8; 4],
        policies: [f16; 4],
        regrets: [f32; 4],
    },
    Large(Box<LargeBucket>),
}

/// Heap storage for buckets with more than 4 edges
struct LargeBucket {
    edges: Vec<u8>,
    policies: Vec<f16>,
    regrets: Vec<f32>,
}

impl LargeBucket {
    fn with_capacity(cap: usize) -> Self {
        Self {
            edges: Vec::with_capacity(cap),
            policies: Vec::with_capacity(cap),
            regrets: Vec::with_capacity(cap),
        }
    }

    fn push_triple(&mut self, edge: u8, policy: f16, regret: f32) {
        self.edges.push(edge);
        self.policies.push(policy);
        self.regrets.push(regret);
    }

    fn len(&self) -> usize {
        self.edges.len()
    }
}

impl CompactBucket {
    /// Create a new empty bucket
    pub fn new() -> Self {
        Self {
            inner: CompactBucketInner::Small {
                count: 0,
                edges: [0; 4],
                policies: [f16::from_f32(0.0); 4],
                regrets: [0.0; 4],
            },
        }
    }

    /// Create a new bucket with specified capacity
    pub fn with_capacity(cap: usize) -> Self {
        if cap <= 4 {
            Self::new()
        } else {
            Self {
                inner: CompactBucketInner::Large(Box::new(LargeBucket::with_capacity(cap))),
            }
        }
    }

    /// Push a new entry (matches SmallVec::push)
    pub fn push(&mut self, entry: (u8, (f16, f32))) {
        let (edge, (policy, regret)) = entry;
        // Always store edge with mask cleared (only bottom 4 bits)
        let edge_val = edge & EDGE_MASK;

        match &mut self.inner {
            CompactBucketInner::Small {
                count,
                edges,
                policies,
                regrets,
            } => {
                // Check if edge already exists
                for i in 0..*count as usize {
                    if (edges[i] & EDGE_MASK) == edge_val {
                        // Update existing, preserve skip bits
                        let skip_bits = edges[i] & SKIP_MASK;
                        edges[i] = edge_val | skip_bits;
                        policies[i] = policy;
                        regrets[i] = regret;
                        return;
                    }
                }

                // Add new edge
                if (*count as usize) < 4 {
                    let idx = *count as usize;
                    edges[idx] = edge_val; // No skip bits for new edge
                    policies[idx] = policy;
                    regrets[idx] = regret;
                    *count += 1;
                } else {
                    // Grow to Large
                    let mut large = Box::new(LargeBucket::with_capacity(5));
                    for i in 0..4 {
                        // Preserve both edge value and skip bits when moving to Large
                        large.push_triple(edges[i], policies[i], regrets[i]);
                    }
                    large.push_triple(edge_val, policy, regret);
                    self.inner = CompactBucketInner::Large(large);
                }
            }
            CompactBucketInner::Large(bucket) => {
                // Check if edge already exists
                if let Some(pos) = bucket.edges.iter().position(|&e| (e & EDGE_MASK) == edge_val) {
                    // Preserve skip bits
                    let skip_bits = bucket.edges[pos] & SKIP_MASK;
                    bucket.edges[pos] = edge_val | skip_bits;
                    bucket.policies[pos] = policy;
                    bucket.regrets[pos] = regret;
                } else {
                    bucket.push_triple(edge_val, policy, regret);
                }
            }
        }
    }

    /// Get length (matches SmallVec::len)
    pub fn len(&self) -> usize {
        match &self.inner {
            CompactBucketInner::Small { count, .. } => *count as usize,
            CompactBucketInner::Large(bucket) => bucket.len(),
        }
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clear all entries
    pub fn clear(&mut self) {
        match &mut self.inner {
            CompactBucketInner::Small { count, .. } => *count = 0,
            CompactBucketInner::Large(bucket) => {
                bucket.edges.clear();
                bucket.policies.clear();
                bucket.regrets.clear();
            }
        }
    }

    /// Iterate over entries (matches SmallVec::iter)
    pub fn iter(&self) -> BucketIter {
        BucketIter {
            bucket: self,
            index: 0,
        }
    }

    /// Iterate over active (non-skipped) entries based on current iteration
    pub fn iter_active(&self, current_iter: u64) -> ActiveBucketIter {
        ActiveBucketIter {
            bucket: self,
            index: 0,
            current_iter,
        }
    }

    /// Mutable iterator
    pub fn iter_mut(&mut self) -> BucketIterMut {
        BucketIterMut {
            bucket: self,
            index: 0,
        }
    }

    /// Find entry by edge (commonly used pattern)
    pub fn find_by_edge(&self, edge: u8) -> Option<(f16, f32)> {
        let edge_val = edge & EDGE_MASK;

        match &self.inner {
            CompactBucketInner::Small {
                count,
                edges,
                policies,
                regrets,
            } => {
                // Unrolled for common cases
                let count = *count as usize;
                if count > 0 && (edges[0] & EDGE_MASK) == edge_val {
                    return Some((policies[0], regrets[0]));
                }
                if count > 1 && (edges[1] & EDGE_MASK) == edge_val {
                    return Some((policies[1], regrets[1]));
                }
                if count > 2 && (edges[2] & EDGE_MASK) == edge_val {
                    return Some((policies[2], regrets[2]));
                }
                if count > 3 && (edges[3] & EDGE_MASK) == edge_val {
                    return Some((policies[3], regrets[3]));
                }
                None
            }
            CompactBucketInner::Large(bucket) => {
                bucket.edges
                    .iter()
                    .position(|&e| (e & EDGE_MASK) == edge_val)
                    .map(|i| (bucket.policies[i], bucket.regrets[i]))
            }
        }
    }

    /// Get policy for edge (returns 0.0 if not found)
    pub fn get_policy(&self, edge: u8) -> f32 {
        self.find_by_edge(edge)
            .map(|(p, _)| f32::from(p))
            .unwrap_or(0.0)
    }

    /// Get regret for edge (returns 0.0 if not found)
    pub fn get_regret(&self, edge: u8) -> f32 {
        self.find_by_edge(edge).map(|(_, r)| r).unwrap_or(0.0)
    }

    /// Update or insert the regret for an edge
    pub fn update_regret<F>(&mut self, edge: u8, updater: F) -> bool
    where
        F: FnOnce(f32) -> f32,
    {
        match &mut self.inner {
            CompactBucketInner::Small {
                count,
                edges,
                policies: _,
                regrets,
            } => {
                let count_val = *count as usize;
                for i in 0..count_val {
                    if edges[i] & EDGE_MASK == edge & EDGE_MASK {
                        regrets[i] = updater(regrets[i]);
                        return true;
                    }
                }
                false
            }
            CompactBucketInner::Large(bucket) => {
                if let Some(pos) = bucket.edges.iter().position(|&e| (e & EDGE_MASK) == (edge & EDGE_MASK)) {
                    bucket.regrets[pos] = updater(bucket.regrets[pos]);
                    true
                } else {
                    false
                }
            }
        }
    }

    /// Update regret and return the new value
    pub fn update_regret_with_value<F>(&mut self, edge: u8, updater: F) -> Option<f32>
    where
        F: FnOnce(f32) -> f32,
    {
        match &mut self.inner {
            CompactBucketInner::Small {
                count,
                edges,
                policies: _,
                regrets,
            } => {
                let count_val = *count as usize;
                for i in 0..count_val {
                    if edges[i] & EDGE_MASK == edge & EDGE_MASK {
                        regrets[i] = updater(regrets[i]);
                        return Some(regrets[i]);
                    }
                }
                None
            }
            CompactBucketInner::Large(bucket) => {
                if let Some(pos) = bucket.edges.iter().position(|&e| (e & EDGE_MASK) == (edge & EDGE_MASK)) {
                    bucket.regrets[pos] = updater(bucket.regrets[pos]);
                    Some(bucket.regrets[pos])
                } else {
                    None
                }
            }
        }
    }

    /// Update or insert the policy for an edge
    pub fn update_policy<F>(&mut self, edge: u8, updater: F) -> bool
    where
        F: FnOnce(f32) -> f32,
    {
        match &mut self.inner {
            CompactBucketInner::Small {
                count,
                edges,
                policies,
                regrets: _,
            } => {
                let count_val = *count as usize;
                for i in 0..count_val {
                    if edges[i] & EDGE_MASK == edge & EDGE_MASK {
                        let old_val = f32::from(policies[i]);
                        policies[i] = f16::from_f32(updater(old_val));
                        return true;
                    }
                }
                false
            }
            CompactBucketInner::Large(bucket) => {
                if let Some(pos) = bucket.edges.iter().position(|&e| (e & EDGE_MASK) == (edge & EDGE_MASK)) {
                    let old_val = f32::from(bucket.policies[pos]);
                    bucket.policies[pos] = f16::from_f32(updater(old_val));
                    true
                } else {
                    false
                }
            }
        }
    }

    /// Calculate sum of positive regrets
    #[inline]
    pub fn sum_positive_regrets(&self) -> f32 {
        match &self.inner {
            CompactBucketInner::Small { count, edges, regrets, .. } => {
                let mut sum = 0.0f32;
                for i in 0..*count as usize {
                    // Skip tombstoned edges
                    if (edges[i] >> 4) != SKIP_LEVEL_TOMBSTONE && regrets[i] > 0.0 {
                        sum += regrets[i];
                    }
                }
                sum
            }
            CompactBucketInner::Large(bucket) => {
                bucket.edges.iter()
                    .zip(bucket.regrets.iter())
                    .filter(|&(&e, &r)| (e >> 4) != SKIP_LEVEL_TOMBSTONE && r > 0.0)
                    .map(|(_, &r)| r)
                    .sum()
            }
        }
    }

    /// Check if an edge is currently skipped
    #[inline]
    pub fn is_edge_skipped(&self, edge: u8, current_iter: u64) -> bool {
        let edge_val = edge & EDGE_MASK;

        match &self.inner {
            CompactBucketInner::Small { count, edges, .. } => {
                for i in 0..*count as usize {
                    if (edges[i] & EDGE_MASK) == edge_val {
                        let skip_level = edges[i] >> 4;
                        if skip_level == SKIP_LEVEL_ACTIVE {
                            return false; // Not skipped
                        }
                        if skip_level == SKIP_LEVEL_TOMBSTONE {
                            return true; // Tombstoned = always skipped
                        }
                        // Check if we haven't reached the skip threshold yet
                        return current_iter < SKIP_ITERATIONS[skip_level as usize] as u64;
                    }
                }
                false // Edge not found = not skipped
            }
            CompactBucketInner::Large(bucket) => {
                if let Some(idx) = bucket.edges.iter().position(|&e| (e & EDGE_MASK) == edge_val) {
                    let skip_level = bucket.edges[idx] >> 4;
                    if skip_level == SKIP_LEVEL_ACTIVE {
                        return false;
                    }
                    if skip_level == SKIP_LEVEL_TOMBSTONE {
                        return true; // Tombstoned = always skipped
                    }
                    // Check if we haven't reached the skip threshold yet
                    return current_iter < SKIP_ITERATIONS[skip_level as usize] as u64;
                }
                false
            }
        }
    }

    /// Count the number of tombstoned edges in this bucket
    pub fn count_tombstoned(&self) -> usize {
        match &self.inner {
            CompactBucketInner::Small { count, edges, .. } => {
                let mut tombstoned = 0;
                for i in 0..*count as usize {
                    if (edges[i] >> 4) == SKIP_LEVEL_TOMBSTONE {
                        tombstoned += 1;
                    }
                }
                tombstoned
            }
            CompactBucketInner::Large(bucket) => {
                bucket.edges.iter()
                    .filter(|&&e| (e >> 4) == SKIP_LEVEL_TOMBSTONE)
                    .count()
            }
        }
    }

    /// Check if an edge is tombstoned (permanently pruned)
    #[inline]
    pub fn is_tombstoned(&self, edge: u8) -> bool {
        let edge_val = edge & EDGE_MASK;

        match &self.inner {
            CompactBucketInner::Small { count, edges, .. } => {
                for i in 0..*count as usize {
                    if (edges[i] & EDGE_MASK) == edge_val {
                        return (edges[i] >> 4) == SKIP_LEVEL_TOMBSTONE;
                    }
                }
                false
            }
            CompactBucketInner::Large(bucket) => {
                bucket.edges.iter()
                    .find(|&&e| (e & EDGE_MASK) == edge_val)
                    .map_or(false, |&e| (e >> 4) == SKIP_LEVEL_TOMBSTONE)
            }
        }
    }

    /// Clear skip status for an edge (when it wakes up)
    pub fn clear_skip(&mut self, edge: u8) {
        let edge_val = edge & EDGE_MASK;

        match &mut self.inner {
            CompactBucketInner::Small { count, edges, .. } => {
                for i in 0..*count as usize {
                    if (edges[i] & EDGE_MASK) == edge_val {
                        edges[i] = edge_val; // Clear skip bits
                        return;
                    }
                }
            }
            CompactBucketInner::Large(bucket) => {
                if let Some(idx) = bucket.edges.iter().position(|&e| (e & EDGE_MASK) == edge_val) {
                    bucket.edges[idx] = edge_val; // Clear skip bits
                }
            }
        }
    }

    /// Check if bucket has any edges with skip bits set
    #[inline]
    pub fn has_skipped_edges(&self) -> bool {
        match &self.inner {
            CompactBucketInner::Small { count, edges, .. } => {
                for i in 0..*count as usize {
                    if edges[i] & SKIP_MASK != 0 {
                        return true;
                    }
                }
                false
            }
            CompactBucketInner::Large(bucket) => {
                bucket.edges.iter().any(|&e| e & SKIP_MASK != 0)
            }
        }
    }

    /// Get edge value and skip level for debugging
    #[allow(dead_code)]
    pub fn edge_skip_info(&self, edge: u8) -> Option<(u8, u8)> {
        let edge_val = edge & EDGE_MASK;

        match &self.inner {
            CompactBucketInner::Small { count, edges, .. } => {
                for i in 0..*count as usize {
                    if (edges[i] & EDGE_MASK) == edge_val {
                        return Some((edges[i] & EDGE_MASK, edges[i] >> 4));
                    }
                }
                None
            }
            CompactBucketInner::Large(bucket) => {
                bucket.edges.iter()
                    .position(|&e| (e & EDGE_MASK) == edge_val)
                    .map(|i| (bucket.edges[i] & EDGE_MASK, bucket.edges[i] >> 4))
            }
        }
    }

    /// Apply RBP check to a single edge after regret update
    /// Returns true if edge was pruned
    ///
    /// current_iter: The number of tree traversals in the CURRENT training run only
    ///               (not cumulative across all runs) to ensure RBP restarts fresh each run
    #[inline]
    pub fn check_and_apply_rbp(&mut self, edge: u8, new_regret: f32, current_iter: u64, horizon: u64) -> bool {
        if new_regret >= 0.0 {
            return false; // Only prune negative regrets
        }

        let pos_sum = self.sum_positive_regrets();
        if pos_sum <= 0.0 {
            return false; // Can't prune if no positive regrets
        }

        let skip_iters = ((-new_regret) / pos_sum).ceil() as u64;

        if current_iter + skip_iters >= horizon {
            // Tombstone
            self.set_skip_level(edge, SKIP_LEVEL_TOMBSTONE);
            true
        } else {
            // Set skip level
            let level = encode_skip_until_iter(current_iter, skip_iters);
            if level > SKIP_LEVEL_ACTIVE {
                self.set_skip_level(edge, level);
                true
            } else {
                false
            }
        }
    }

    /// Set skip level for an edge
    #[inline]
    fn set_skip_level(&mut self, edge: u8, level: u8) {
        let edge_val = edge & EDGE_MASK;

        match &mut self.inner {
            CompactBucketInner::Small { count, edges, .. } => {
                for i in 0..*count as usize {
                    if (edges[i] & EDGE_MASK) == edge_val {
                        edges[i] = edge_val | (level << 4);
                        return;
                    }
                }
            }
            CompactBucketInner::Large(bucket) => {
                if let Some(idx) = bucket.edges.iter().position(|&e| (e & EDGE_MASK) == edge_val) {
                    bucket.edges[idx] = edge_val | (level << 4);
                }
            }
        }
    }
}

impl Default for CompactBucket {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for CompactBucket {
    fn clone(&self) -> Self {
        match &self.inner {
            CompactBucketInner::Small {
                count,
                edges,
                policies,
                regrets,
            } => Self {
                inner: CompactBucketInner::Small {
                    count: *count,
                    edges: *edges,
                    policies: *policies,
                    regrets: *regrets,
                },
            },
            CompactBucketInner::Large(bucket) => Self {
                inner: CompactBucketInner::Large(Box::new(LargeBucket {
                    edges: bucket.edges.clone(),
                    policies: bucket.policies.clone(),
                    regrets: bucket.regrets.clone(),
                })),
            },
        }
    }
}

/// Iterator over bucket entries
pub struct BucketIter<'a> {
    bucket: &'a CompactBucket,
    index: usize,
}

impl<'a> Iterator for BucketIter<'a> {
    type Item = (u8, (f16, f32));

    fn next(&mut self) -> Option<Self::Item> {
        match &self.bucket.inner {
            CompactBucketInner::Small {
                count,
                edges,
                policies,
                regrets,
            } => {
                while self.index < *count as usize {
                    let i = self.index;
                    self.index += 1;

                    // Skip tombstoned edges (permanently pruned)
                    if (edges[i] >> 4) == SKIP_LEVEL_TOMBSTONE {
                        continue;
                    }

                    // Return edge value without skip bits
                    return Some((edges[i] & EDGE_MASK, (policies[i], regrets[i])));
                }
                None
            }
            CompactBucketInner::Large(bucket) => {
                while self.index < bucket.edges.len() {
                    let i = self.index;
                    self.index += 1;

                    // Skip tombstoned edges (permanently pruned)
                    if (bucket.edges[i] >> 4) == SKIP_LEVEL_TOMBSTONE {
                        continue;
                    }

                    // Return edge value without skip bits
                    return Some((bucket.edges[i] & EDGE_MASK, (bucket.policies[i], bucket.regrets[i])));
                }
                None
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.bucket.len() - self.index;
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for BucketIter<'a> {}

/// Mutable iterator over bucket entries
pub struct BucketIterMut<'a> {
    bucket: &'a mut CompactBucket,
    index: usize,
}

impl<'a> Iterator for BucketIterMut<'a> {
    type Item = (&'a mut u8, &'a mut f16, &'a mut f32);

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.bucket.len() {
            return None;
        }

        let index = self.index;
        self.index += 1;

        // Use unsafe to return mutable references with the correct lifetime
        unsafe {
            match &mut (*(&mut self.bucket.inner as *mut CompactBucketInner)) {
                CompactBucketInner::Small {
                    count,
                    edges,
                    policies,
                    regrets,
                } => {
                    if index < *count as usize {
                        Some((
                            &mut *(edges.as_mut_ptr().add(index)),
                            &mut *(policies.as_mut_ptr().add(index)),
                            &mut *(regrets.as_mut_ptr().add(index)),
                        ))
                    } else {
                        None
                    }
                }
                CompactBucketInner::Large(bucket) => {
                    let bucket = &mut **bucket;
                    if index < bucket.len() {
                        Some((
                            &mut *(bucket.edges.as_mut_ptr().add(index)),
                            &mut *(bucket.policies.as_mut_ptr().add(index)),
                            &mut *(bucket.regrets.as_mut_ptr().add(index)),
                        ))
                    } else {
                        None
                    }
                }
            }
        }
    }
}

/// Iterator over active (non-skipped) entries
pub struct ActiveBucketIter<'a> {
    bucket: &'a CompactBucket,
    index: usize,
    current_iter: u64,
}

impl<'a> Iterator for ActiveBucketIter<'a> {
    type Item = (u8, (f16, f32));

    fn next(&mut self) -> Option<Self::Item> {
        match &self.bucket.inner {
            CompactBucketInner::Small {
                count,
                edges,
                policies,
                regrets,
            } => {
                while self.index < *count as usize {
                    let i = self.index;
                    self.index += 1;

                    let skip_level = edges[i] >> 4;
                    // Skip if pruned (tombstoned or temporarily skipped)
                    if skip_level == SKIP_LEVEL_TOMBSTONE {
                        continue;
                    }
                    if skip_level != SKIP_LEVEL_ACTIVE &&
                       self.current_iter < SKIP_ITERATIONS[skip_level as usize] as u64 {
                        continue;
                    }

                    // Return edge value without skip bits
                    return Some((edges[i] & EDGE_MASK, (policies[i], regrets[i])));
                }
                None
            }
            CompactBucketInner::Large(bucket) => {
                while self.index < bucket.edges.len() {
                    let i = self.index;
                    self.index += 1;

                    let skip_level = bucket.edges[i] >> 4;
                    // Skip if pruned (tombstoned or temporarily skipped)
                    if skip_level == SKIP_LEVEL_TOMBSTONE {
                        continue;
                    }
                    if skip_level != SKIP_LEVEL_ACTIVE &&
                       self.current_iter < SKIP_ITERATIONS[skip_level as usize] as u64 {
                        continue;
                    }

                    // Return edge value without skip bits
                    return Some((bucket.edges[i] & EDGE_MASK, (bucket.policies[i], bucket.regrets[i])));
                }
                None
            }
        }
    }
}

impl fmt::Debug for CompactBucket {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CompactBucket[")?;
        for (i, (edge, (policy, regret))) in self.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "({}, ({}, {}))", edge, f32::from(policy), regret)?;
        }
        write!(f, "]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: f16 (half-precision float) has limited precision.
    // When converting f32 -> f16 -> f32, we lose precision.
    // For example: 0.2 becomes 0.19995117, 0.7 becomes 0.7001953
    // This is expected and part of the memory/precision tradeoff.
    // Tests use approximate equality (< 0.001 tolerance) for f16 values.

    #[test]
    fn test_small_bucket() {
        let mut bucket = CompactBucket::new();
        assert_eq!(bucket.len(), 0);

        bucket.push((1, (f16::from_f32(0.5), 1.0)));
        bucket.push((2, (f16::from_f32(0.3), -0.5)));

        assert_eq!(bucket.len(), 2);
        assert_eq!(bucket.get_policy(1), 0.5);
        assert_eq!(bucket.get_regret(2), -0.5);
        assert_eq!(bucket.get_policy(3), 0.0);
    }

    #[test]
    fn test_grow_to_large() {
        let mut bucket = CompactBucket::new();

        // Fill small bucket
        for i in 0..4 {
            bucket.push((i, (f16::from_f32(i as f32), i as f32 * 2.0)));
        }

        // This should trigger growth
        bucket.push((4, (f16::from_f32(4.0), 8.0)));

        assert_eq!(bucket.len(), 5);
        // Use approximate equality for f16 precision
        assert!((bucket.get_policy(4) - 4.0).abs() < 0.001);
    }

    #[test]
    fn test_update_existing() {
        let mut bucket = CompactBucket::new();

        bucket.push((1, (f16::from_f32(0.5), 1.0)));
        bucket.push((1, (f16::from_f32(0.7), 2.0))); // Update

        assert_eq!(bucket.len(), 1);
        // Use approximate equality for f16 precision
        assert!((bucket.get_policy(1) - 0.7).abs() < 0.001);
        assert_eq!(bucket.get_regret(1), 2.0);
    }

    #[test]
    fn test_iter() {
        let mut bucket = CompactBucket::new();
        bucket.push((1, (f16::from_f32(0.1), 1.0)));
        bucket.push((2, (f16::from_f32(0.2), 2.0)));

        let collected: Vec<_> = bucket.iter().collect();
        assert_eq!(collected.len(), 2);
        assert_eq!(collected[0].0, 1);
        // Use approximate equality for f16 precision
        assert!((f32::from(collected[1].1 .0) - 0.2).abs() < 0.001);
    }

    #[test]
    fn test_memory_size() {
        use std::mem::size_of;

        // Verify our memory assumptions
        assert_eq!(size_of::<CompactBucket>(), 32);
        assert_eq!(size_of::<CompactBucketInner>(), 32);
    }

    #[test]
    fn test_memory_comparison() {
        use smallvec::SmallVec;
        use std::mem::size_of;

        // Original SmallVec type
        type OriginalBucket = SmallVec<[(u8, (f16, f32)); 4]>;

        println!("Memory size comparison:");
        println!(
            "  SmallVec<[(u8, (f16, f32)); 4]>: {} bytes",
            size_of::<OriginalBucket>()
        );
        println!("  CompactBucket: {} bytes", size_of::<CompactBucket>());
        println!(
            "  Savings: {} bytes ({:.1}% reduction)",
            size_of::<OriginalBucket>() - size_of::<CompactBucket>(),
            (1.0 - size_of::<CompactBucket>() as f64 / size_of::<OriginalBucket>() as f64) * 100.0
        );

        // CompactBucket should be significantly smaller
        assert!(size_of::<CompactBucket>() < size_of::<OriginalBucket>());
    }

    #[test]
    fn test_stable_rbp() {
        let mut bucket = CompactBucket::new();

        // Add edges with regrets
        bucket.push((1, (f16::from_f32(0.5), 100.0)));  // Positive regret
        bucket.push((2, (f16::from_f32(0.3), -50.0)));  // Negative regret
        bucket.push((3, (f16::from_f32(0.2), 25.0)));   // Positive regret

        // Apply RBP pruning to each edge manually
        let current_iter = 1_000_000;
        let horizon = 0x2000000; // 33.5M
        bucket.check_and_apply_rbp(2, -50.0, current_iter, horizon);

        // Edge 2: skip_iters = ceil(50/125) = 1
        // current_iter (1M) + skip_iters (1) = 1,000,001
        // Since 1,000,001 < 2,097,152 (first threshold), no skip is encoded
        assert!(!bucket.is_edge_skipped(2, 1_000_000));   // Should NOT be skipped (skip too small)

        // Let's test with a larger negative regret that would actually trigger skipping
        bucket.push((4, (f16::from_f32(0.1), -1_000_000.0)));  // Very negative
        bucket.check_and_apply_rbp(4, -1_000_000.0, current_iter, horizon);
        // skip_iters = ceil(1M/125) = 8000
        // target = 1M + 8000 = 1,008,000 < 2.1M, so still no skip
        assert!(!bucket.is_edge_skipped(4, current_iter));

        // Edge 1 should not be skipped (positive regret)
        assert!(!bucket.is_edge_skipped(1, current_iter));
        // Edge 3 should not be skipped (positive regret, no RBP applied)
        assert!(!bucket.is_edge_skipped(3, current_iter));

        // Test iter_active - all edges should be active since skip_iters were too small
        let active: Vec<u8> = bucket.iter_active(current_iter).map(|(e, _)| e).collect();
        assert_eq!(active, vec![1, 2, 3, 4]); // All edges active (no meaningful skips)
    }

    #[test]
    fn test_rbp_large_negative() {
        let mut bucket = CompactBucket::new();

        // Edge with very negative regret
        bucket.push((1, (f16::from_f32(0.5), 10.0)));     // Small positive
        bucket.push((2, (f16::from_f32(0.3), -10000.0))); // Large negative

        // Apply RBP check
        bucket.check_and_apply_rbp(2, -10000.0, 0, 50_000_000);

        // skip_iters = ceil(10000/10) = 1000
        // target = 0 + 1000 = 1000
        // Since 1000 < 2,097,152 (first threshold), no skip is encoded
        let skip_info = bucket.edge_skip_info(2);
        assert!(skip_info.is_some());
        let (_, level) = skip_info.unwrap();
        assert_eq!(level, SKIP_LEVEL_ACTIVE); // No skip for small skip_iters

        // Create a scenario where skip_iters would be large enough
        let mut bucket2 = CompactBucket::new();
        bucket2.push((1, (f16::from_f32(0.1), 0.001)));    // Tiny positive regret
        bucket2.push((2, (f16::from_f32(0.1), -3000.0)));  // Large negative

        // skip_iters = ceil(3000/0.001) = 3,000,000
        // target = 0 + 3,000,000 = 3,000,000
        // This is between level 1 (2.1M) and level 2 (4.2M)
        // So it should get level 1 (skip until 2.1M)
        bucket2.check_and_apply_rbp(2, -3000.0, 0, crate::CFR_TREE_COUNT_NLHE as u64);
        let skip_info2 = bucket2.edge_skip_info(2);
        assert_eq!(skip_info2, Some((2, 1))); // Level 1
        assert!(bucket2.is_edge_skipped(2, 0));
        assert!(bucket2.is_edge_skipped(2, 2_000_000));
        assert!(!bucket2.is_edge_skipped(2, 2_100_000)); // Wake up after threshold
    }

    #[test]
    fn test_rbp_integration() {
        let mut bucket = CompactBucket::new();

        // Add edges with regrets
        bucket.push((1, (f16::from_f32(0.5), 50.0)));   // Positive
        bucket.push((2, (f16::from_f32(0.3), -25.0)));  // Negative
        bucket.push((3, (f16::from_f32(0.2), -1000.0))); // Very negative

        // Apply RBP to negative regret edges
        bucket.check_and_apply_rbp(2, -25.0, 0, 50_000_000);
        bucket.check_and_apply_rbp(3, -1000.0, 0, 50_000_000);

        // Test iter() skips tombstoned edges
        let active_edges: Vec<u8> = bucket.iter().map(|(e, _)| e).collect();
        assert_eq!(active_edges.len(), 3); // All still visible in basic iter

        // Test that normal get_regret still returns the actual values
        assert_eq!(bucket.get_regret(1), 50.0);   // Active
        assert_eq!(bucket.get_regret(2), -25.0);  // Still has negative regret
        assert_eq!(bucket.get_regret(3), -1000.0); // Still has negative regret

        // Test sum of positive regrets still only counts non-tombstoned
        assert_eq!(bucket.sum_positive_regrets(), 50.0);

        // Manually tombstone edge 3
        if let CompactBucketInner::Small { edges, .. } = &mut bucket.inner {
            edges[2] = (edges[2] & EDGE_MASK) | (SKIP_LEVEL_TOMBSTONE << 4);
        }

        // Now iter() should skip the tombstoned edge
        let active_edges: Vec<u8> = bucket.iter().map(|(e, _)| e).collect();
        assert_eq!(active_edges, vec![1, 2]); // Edge 3 is hidden
    }

    #[test]
    fn test_rbp_warmup_respected() {
        // This test ensures that RBP checks are not applied during warmup period
        // The actual warmup check is in blueprint.rs, but we can test the mechanism here
        let mut bucket = CompactBucket::new();

        // Add edges with regrets
        bucket.push((1, (f16::from_f32(0.5), 100.0)));  // Positive
        bucket.push((2, (f16::from_f32(0.3), -50.0)));  // Negative

        // With skip_iters = ceil(50/100) = 1, target = 1001
        // Since 1001 < 2,097,152, no pruning occurs (skip too small)
        let was_pruned = bucket.check_and_apply_rbp(2, -50.0, 1000, 50_000_000);
        assert!(!was_pruned); // Skip too small to encode

        // Test with larger negative regret that would actually prune
        bucket.push((3, (f16::from_f32(0.1), 0.01)));     // Tiny positive
        bucket.push((4, (f16::from_f32(0.1), -2100.0)));  // Large negative

        // skip_iters = ceil(2100/100.01) = 21, but we need millions to reach first threshold
        // Let's use an extreme case
        let mut bucket2 = CompactBucket::new();
        bucket2.push((1, (f16::from_f32(0.1), 0.001)));   // Tiny positive
        bucket2.push((2, (f16::from_f32(0.1), -2100.0))); // Large negative

        // skip_iters = ceil(2100/0.001) = 2,100,000
        // target = 0 + 2,100,000 = 2,100,000 > 2,097,152 (first threshold)
        let was_pruned2 = bucket2.check_and_apply_rbp(2, -2100.0, 0, crate::CFR_TREE_COUNT_NLHE as u64);
        assert!(was_pruned2); // This one should prune
        assert!(bucket2.is_edge_skipped(2, 0));
    }
}
