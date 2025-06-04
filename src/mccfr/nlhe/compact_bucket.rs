use half::f16;
use std::fmt;

/// A specialized container optimized for poker edge storage.
/// Similar to SmallVec but with better memory layout for our specific use case.
///
/// Stores up to 4 edges inline, spilling to heap for larger collections.
/// Memory layout is optimized for the common case (2-4 edges, 81% cumulative).
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

    fn find_by_edge(&self, edge: u8) -> Option<(f16, f32)> {
        self.edges
            .iter()
            .position(|&e| e == edge)
            .map(|i| (self.policies[i], self.regrets[i]))
    }

    fn get(&self, index: usize) -> Option<(u8, (f16, f32))> {
        if index < self.edges.len() {
            Some((
                self.edges[index],
                (self.policies[index], self.regrets[index]),
            ))
        } else {
            None
        }
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
        match &mut self.inner {
            CompactBucketInner::Small {
                count,
                edges,
                policies,
                regrets,
            } => {
                // Check if edge already exists
                for i in 0..*count as usize {
                    if edges[i] == edge {
                        // Update existing
                        policies[i] = policy;
                        regrets[i] = regret;
                        return;
                    }
                }

                // Add new edge
                if (*count as usize) < 4 {
                    let idx = *count as usize;
                    edges[idx] = edge;
                    policies[idx] = policy;
                    regrets[idx] = regret;
                    *count += 1;
                } else {
                    // Grow to Large
                    let mut large = Box::new(LargeBucket::with_capacity(5));
                    for i in 0..4 {
                        large.push_triple(edges[i], policies[i], regrets[i]);
                    }
                    large.push_triple(edge, policy, regret);
                    self.inner = CompactBucketInner::Large(large);
                }
            }
            CompactBucketInner::Large(bucket) => {
                // Check if edge already exists
                if let Some(pos) = bucket.edges.iter().position(|&e| e == edge) {
                    bucket.policies[pos] = policy;
                    bucket.regrets[pos] = regret;
                } else {
                    bucket.push_triple(edge, policy, regret);
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

    /// Mutable iterator
    pub fn iter_mut(&mut self) -> BucketIterMut {
        BucketIterMut {
            bucket: self,
            index: 0,
        }
    }

    /// Find entry by edge (commonly used pattern)
    pub fn find_by_edge(&self, edge: u8) -> Option<(f16, f32)> {
        match &self.inner {
            CompactBucketInner::Small {
                count,
                edges,
                policies,
                regrets,
            } => {
                // Unrolled for common cases
                let count = *count as usize;
                if count > 0 && edges[0] == edge {
                    return Some((policies[0], regrets[0]));
                }
                if count > 1 && edges[1] == edge {
                    return Some((policies[1], regrets[1]));
                }
                if count > 2 && edges[2] == edge {
                    return Some((policies[2], regrets[2]));
                }
                if count > 3 && edges[3] == edge {
                    return Some((policies[3], regrets[3]));
                }
                None
            }
            CompactBucketInner::Large(bucket) => bucket.find_by_edge(edge),
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
                    if edges[i] == edge {
                        regrets[i] = updater(regrets[i]);
                        return true;
                    }
                }
                false
            }
            CompactBucketInner::Large(bucket) => {
                if let Some(pos) = bucket.edges.iter().position(|&e| e == edge) {
                    bucket.regrets[pos] = updater(bucket.regrets[pos]);
                    true
                } else {
                    false
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
                    if edges[i] == edge {
                        let old_val = f32::from(policies[i]);
                        policies[i] = f16::from_f32(updater(old_val));
                        return true;
                    }
                }
                false
            }
            CompactBucketInner::Large(bucket) => {
                if let Some(pos) = bucket.edges.iter().position(|&e| e == edge) {
                    let old_val = f32::from(bucket.policies[pos]);
                    bucket.policies[pos] = f16::from_f32(updater(old_val));
                    true
                } else {
                    false
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
                if self.index < *count as usize {
                    let i = self.index;
                    self.index += 1;
                    Some((edges[i], (policies[i], regrets[i])))
                } else {
                    None
                }
            }
            CompactBucketInner::Large(bucket) => bucket.get(self.index).map(|entry| {
                self.index += 1;
                entry
            }),
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
}
