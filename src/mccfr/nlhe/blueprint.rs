use crate::mccfr::traits::blueprint::Blueprint;
use crate::mccfr::traits::profile::Profile;

impl Blueprint for super::solver::NLHE {
    type T = super::turn::Turn;
    type E = super::edge::Edge;
    type G = super::game::Game;
    type I = super::info::Info;
    type P = super::profile::Profile;
    type S = super::encoder::Encoder;

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
        use rustc_hash::FxHashMap;
        #[cfg(feature = "native")]
        use crate::save::disk::Disk;

        let total_iters = <Self as crate::mccfr::traits::blueprint::Blueprint>::iterations();

        const SAVE_TARGET_RAW: usize = 250_000;
        let save_interval = (SAVE_TARGET_RAW / Self::batch_size()).max(1);

        log::info!("Starting training: {} iterations", total_iters);

        // Initialize progress bar (native targets only). The long tick interval ensures
        // updates are throttled and do not add noticeable overhead to the solver.
        #[cfg(feature = "native")]
        let progress = crate::progress(total_iters);

        'training: for i in 0..total_iters {
            #[cfg(feature = "native")]
            progress.inc(1);

            // Periodic checkpoint
            if (i + 1) % save_interval == 0 {
                #[cfg(feature = "native")]
                {
                    self.profile.save();
                    log::info!("Checkpoint saved after {} outer iterations", i + 1);
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
                        // Access the bucket-level map via DashMap + Mutex
                        let bucket_mutex = profile_ref
                            .encounters
                            .entry(info)
                            .or_insert_with(|| parking_lot::Mutex::new(FxHashMap::with_capacity_and_hasher(4, Default::default())));

                        let mut bucket = bucket_mutex.lock();

                        let entry = bucket.entry(edge).or_insert((0.0f32, 0.0f32));

                        // Current values
                        let current_policy = entry.0;
                        let current_regret = entry.1;

                        // Discounts (cached)
                        let discount_r = if current_regret > 0.0 {
                            discount_pos
                        } else if current_regret < 0.0 {
                            discount_neg
                        } else {
                            discount_none
                        };
                        let discount_p = discount_none; // policy discount never depends on sign

                        // Update in-place
                        entry.1 = (current_regret * discount_r + delta_r).max(crate::REGRET_MIN);
                        entry.0 = (current_policy * discount_p + delta_p).max(crate::POLICY_MIN);
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
