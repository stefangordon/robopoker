# Memory Optimization Guide for MCCFR Profile

## Current Memory Usage (9.13B infosets)
- Per infoset: ~112 bytes
- Total: ~1TB RAM

## Optimization Options for 10%+ Savings:

### 1. **Zero-Regret Pruning** (16.93% savings)
Since 16.93% of your regrets are zero, you could skip storing them:
```rust
// Only store non-zero regrets
if regret != 0.0 || policy != 0.0 {
    bucket.push((edge_key, (policy_f16, regret)));
}
```

### 2. **Custom Allocator** (5-10% savings)
Use jemalloc or mimalloc:
```toml
[dependencies]
jemallocator = "0.5"
```

### 3. **Reduce SmallVec Size** (Variable savings)
Based on edge distribution:
- If >50% have ≤3 edges: Use SmallVec<[_; 3]>
- Run with BLUEPRINT_STATS=1 to check distribution

### 4. **Compress During Training** (Memory vs Speed tradeoff)
Store compressed pages, decompress on access:
```rust
use lz4_flex::compress_prepend_size;
// Compress buckets with >N entries
```

### 5. **Sparse Profile Storage** (20-30% savings)
Only store infosets that differ from default:
```rust
// Skip infosets with uniform strategy
if is_uniform_strategy(&bucket) {
    return; // Don't store
}
```

### 6. **Memory-Mapped Profile** (50%+ virtual memory savings)
Use mmap for large profiles:
```rust
use memmap2::MmapOptions;
// Load profile as memory-mapped file
```

## Quick Win: Reduce Parallelism
If you're memory constrained:
```bash
RAYON_NUM_THREADS=4 cargo run --release -- --trainer
```
This reduces peak memory usage during training.

## Recommended Approach:
1. First: Add zero-regret pruning (easy 16.93% win)
2. Second: Use jemalloc (5-10% win)
3. Third: Consider sparse storage for uniform strategies

## Memory Calculation:
Current: 9.13B × 112 bytes = 1.02TB
With optimizations: 9.13B × 94 bytes = 0.86TB (16% savings) 