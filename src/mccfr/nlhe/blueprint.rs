use crate::mccfr::traits::blueprint::Blueprint;
use crate::mccfr::traits::profile::Profile;
use crate::save::disk::Disk;
use rustc_hash::FxHashMap;
use half::f16;

impl Blueprint for super::solver::NLHE {
    type T = super::turn::Turn;
    type E = super::edge::Edge;
    type G = super::game::Game;
    type I = super::info::Info;
    type P = super::profile::Profile;
    type S = super::encoder::BlueprintEncoder;

    fn train() {
        use crate::cards::street::Street;
        use crate::save::disk::Disk;
        use crate::Arbitrary;
        if Self::done(Street::random()) {
            log::info!("resuming regret minimization from checkpoint");
            Self::load(Street::random()).solve().save();
        } else {
            log::info!("starting regret minimization from scratch");
            Self::grow(Street::random()).solve().save();
        }
    }

    fn tree_count() -> usize {
        crate::CFR_TREE_COUNT_NLHE
    }
    fn batch_size() -> usize {
        crate::CFR_BATCH_SIZE_NLHE
    }

    fn advance(&mut self) {
        self.profile.increment();
    }
    fn encoder(&self) -> &Self::S {
        &self.sampler
    }
    fn profile(&self) -> &Self::P {
        &self.profile
    }
    fn mut_policy(&mut self, _info: &Self::I, _edge: &Self::E) -> &mut f32 {
        panic!("mut_policy is unused in concurrent solve implementation");
    }
    fn mut_regret(&mut self, _info: &Self::I, _edge: &Self::E) -> &mut f32 {
        panic!("mut_regret is unused in concurrent solve implementation");
    }

    // Override solve to use concurrent updates into Profile's DashMap, eliminating the
    // serial merge step in the default implementation.
    fn solve(mut self) -> Self {
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        use std::time::{Duration, Instant};

        let total_iters = <Self as crate::mccfr::traits::blueprint::Blueprint>::iterations();

        log::info!("Starting training: {} iterations", total_iters);

        // Time-based checkpoint interval
        #[cfg(feature = "native")]
        let checkpoint_interval = Duration::from_secs(3600 * crate::CHECKPOINT_HOURS);
        #[cfg(feature = "native")]
        let mut last_checkpoint = Instant::now();

        // Initialize progress bar (native targets only). The long tick interval ensures
        // updates are throttled and do not add noticeable overhead to the solver.
        #[cfg(feature = "native")]
        let progress = crate::progress(total_iters);

        'training: for i in 0..total_iters {
            // Periodic checkpoint every CHECKPOINT_HOURS hours
            #[cfg(feature = "native")]
            {
                progress.inc(1);
                if last_checkpoint.elapsed() >= checkpoint_interval {
                    self.profile.save();
                    
                    // Also log stats during checkpoint
                    self.profile.log_stats();
                    
                    last_checkpoint = Instant::now();
                    log::info!(
                        "Checkpoint saved after {} elapsed hours (iteration {})",
                        crate::CHECKPOINT_HOURS,
                        i + 1
                    );
                }
            }

            let walker = self.profile.walker();
            let profile_ref = &self.profile; // immutable, but interior-mutable via DashMap

            // Pre-compute discount factors once per iteration (optimization #3)
            let discount_none = self.discount(None);
            let discount_pos = self.discount(Some(1.0)); // positive regret
            let discount_neg = self.discount(Some(-1.0)); // negative regret

            (0..Self::batch_size())
                .into_par_iter()
                // Build tree
                .map(|_| self.tree())
                // Extract infosets for the walker
                .flat_map_iter(move |tree| {
                    tree.partition()
                        .into_values()
                        .into_iter()
                        .filter(move |infoset| infoset.head().game().turn() == walker)
                })
                // Convert to counterfactual triples
                .map(|infoset| self.counterfactual(infoset))
                // Thread-local aggregation with a pre-reserved map to avoid reallocs (opt #1)
                .fold(
                    || FxHashMap::<(super::info::Info, super::edge::Edge), (f32, f32)>::with_capacity_and_hasher(1024, Default::default()),
                    |mut local, (info, regret_vec, policy_vec)| {
                        for (edge, regret) in regret_vec {
                            let entry = local.entry((info, edge)).or_insert((0.0f32, 0.0f32));
                            entry.1 += regret; // accumulate regret in .1
                        }
                        for (edge, policy) in policy_vec {
                            let entry = local.entry((info, edge)).or_insert((0.0f32, 0.0f32));
                            entry.0 += policy; // accumulate policy in .0
                        }
                        local
                    },
                )
                // Flush each local map into the global profile in parallel
                .for_each(|local| {
                    for ((info, edge), (delta_p, delta_r)) in local.into_iter() {
                        let edge_key = u8::from(edge);
                        // Access the bucket-level map via DashMap + Mutex
                        let bucket_mutex = profile_ref
                            .encounters
                            .entry(info)
                            .or_insert_with(|| parking_lot::RwLock::new(super::compact_bucket::CompactBucket::new()));

                        let mut bucket = bucket_mutex.value().write();

                        // Search for edge record and update
                        if bucket.update_policy(edge_key, |current_policy| {
                            let discount_p = discount_none;
                            (current_policy * discount_p + delta_p).max(crate::POLICY_MIN)
                        }) {
                            // Policy was updated, now update regret
                            bucket.update_regret(edge_key, |current_regret| {
                                let discount_r = if current_regret > 0.0 {
                                    discount_pos
                                } else if current_regret < 0.0 {
                                    discount_neg
                                } else {
                                    discount_none
                                };
                                (current_regret * discount_r + delta_r)
                                    .clamp(crate::REGRET_MIN, crate::REGRET_MAX)
                            });
                        } else {
                            // No existing record; create new using discounts on zero
                            let new_regret = (0.0 * discount_none + delta_r).clamp(crate::REGRET_MIN, crate::REGRET_MAX);
                            let new_policy = (0.0 * discount_none + delta_p).max(crate::POLICY_MIN);
                            #[cfg(debug_assertions)]
                            debug_assert!(
                                bucket.iter().all(|(e, _)| e != edge_key),
                                "duplicate edge {:?} detected in infoset {:?}",
                                edge,
                                info
                            );
                            bucket.push((edge_key, (f16::from_f32(new_policy), new_regret)));
                        }
                    }
                });

            // handle interrupt / advance epoch
            if self.interrupted() {
                break 'training;
            }
        }

        // Finalize progress bar once training ends or is interrupted.
        #[cfg(feature = "native")]
        {
            progress.finish();
            println!();
        }

        self
    }
}
