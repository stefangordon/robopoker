criterion::criterion_main!(benches);
criterion::criterion_group! {
    name = benches;
    config = criterion::Criterion::default()
        .without_plots()
        .noise_threshold(3.0)
        .significance_level(0.01)
        .sample_size(10)
        .measurement_time(std::time::Duration::from_secs(1));
    targets =
        sampling_river_evaluation,
        sampling_river_equity,
        sampling_river_observation,
        converting_turn_isomorphism,
        exhausting_flop_observations,
        exhausting_flop_isomorphisms,
        collecting_turn_histogram,
        computing_optimal_transport_variation,
        computing_optimal_transport_heuristic,
        computing_optimal_transport_sinkhorns,
        solving_cfr_rps,
        saving_profile_blueprint,
}

fn sampling_river_evaluation(c: &mut criterion::Criterion) {
    c.bench_function("evaluate a 7-card Hand", |b| {
        let hand = Hand::from(Observation::from(Street::Rive));
        b.iter(|| Strength::from(Evaluator::from(hand)))
    });
}

fn sampling_river_observation(c: &mut criterion::Criterion) {
    c.bench_function("collect a 7-card River Observation", |b| {
        b.iter(|| Observation::from(Street::Rive))
    });
}

fn sampling_river_equity(c: &mut criterion::Criterion) {
    let observation = Observation::from(Street::Rive);
    c.bench_function("calculate River equity", |b| {
        b.iter(|| observation.equity())
    });
}

fn exhausting_flop_observations(c: &mut criterion::Criterion) {
    c.bench_function("exhaust all Flop Observations", |b| {
        b.iter(|| ObservationIterator::from(Street::Flop).count())
    });
}

fn exhausting_flop_isomorphisms(c: &mut criterion::Criterion) {
    c.bench_function("exhaust all Flop Isomorphisms", |b| {
        b.iter(|| {
            ObservationIterator::from(Street::Flop)
                .filter(Isomorphism::is_canonical)
                .count()
        })
    });
}

fn converting_turn_isomorphism(c: &mut criterion::Criterion) {
    let observation = Observation::from(Street::Turn);
    c.bench_function("convert a Turn Observation to Isomorphism", |b| {
        b.iter(|| Isomorphism::from(observation))
    });
}

fn collecting_turn_histogram(c: &mut criterion::Criterion) {
    let observation = Observation::from(Street::Turn);
    c.bench_function("collect a Histogram from a Turn Observation", |b| {
        b.iter(|| Histogram::from(observation))
    });
}

fn computing_optimal_transport_variation(c: &mut criterion::Criterion) {
    let ref h1 = Histogram::from(Observation::from(Street::Turn));
    let ref h2 = Histogram::from(Observation::from(Street::Turn));
    c.bench_function("compute optimal transport (1-dimensional)", |b| {
        b.iter(|| Equity::variation(&h1, &h2))
    });
}

fn computing_optimal_transport_heuristic(c: &mut criterion::Criterion) {
    let (metric, h1, h2, _) = EMD::random().inner();
    c.bench_function("compute optimal transport (greedy)", |b| {
        b.iter(|| Heuristic::from((&h1, &h2, &metric)).minimize().cost())
    });
}

fn computing_optimal_transport_sinkhorns(c: &mut criterion::Criterion) {
    let (metric, h1, h2, _) = EMD::random().inner();
    c.bench_function("compute optimal transport (entropy regularized)", |b| {
        b.iter(|| Sinkhorn::from((&h1, &h2, &metric)).minimize().cost())
    });
    /*
    TEMPERATURE   ITERS  TOLERANCE  TIME
        0.125       16     0.001     200
        0.125       16     0.010     135
        0.125       16     0.100     67
        8.000       16     0.001     55
        8.000       16     0.010     55
        8.000       16     0.100     55
     */
}

fn solving_cfr_rps(c: &mut criterion::Criterion) {
    c.bench_function("cfr solve rock paper scissors (rps)", |b| {
        b.iter(|| RPS::default().solve());
    });
}

#[cfg(feature = "native")]
fn saving_profile_blueprint(c: &mut criterion::Criterion) {
    use robopoker::cards::street::Street;
    use robopoker::mccfr::nlhe::profile::Profile;
    use robopoker::save::disk::Disk;
    use std::fs;
    use std::path::Path;
    use std::path::PathBuf;
    use std::time::Instant;

    c.bench_function("save profile blueprint", |b| {
        // Create a profile with some data
        // We'll generate it once and then benchmark just the save operation
        let mut profile = Profile::default();

        // Populate profile with synthetic data for benchmarking
        profile.populate_dummy(1_000_000, 4);

        // Create a temporary directory for the save
        let temp_dir =
            PathBuf::from(std::env::var("USERPROFILE").unwrap()).join("pgcopy_benchmark");

        // Ensure all parent directories exist
        fs::create_dir_all(&temp_dir).expect("Failed to create temp directory");
        println!("Benchmark using temp directory: {}", temp_dir.display());

        let old_dir = std::env::current_dir().unwrap();
        std::env::set_current_dir(&temp_dir).expect("Failed to change to temp directory");

        // Create the pgcopy subdirectory which might be needed
        let pgcopy_dir = temp_dir.join("pgcopy");
        fs::create_dir_all(&pgcopy_dir).expect("Failed to create pgcopy directory");
        println!("Created pgcopy directory: {}", pgcopy_dir.display());

        // Verify we can create/measure the test directories
        let _path = Profile::path(Street::random());

        println!("Using path from Profile::path");

        // Since the real function logs progress, let's time it once outside the benchmark
        // to give users an idea of what to expect
        let start = Instant::now();
        profile.save();
        let duration = start.elapsed();
        println!("Initial save took: {:?}", duration);

        // Now benchmark the actual save function
        b.iter(|| {
            profile.save();
        });

        // Clean up
        std::env::set_current_dir(old_dir).unwrap();
        if Path::new(&temp_dir).exists() {
            let _ = fs::remove_dir_all(&temp_dir);
        }
    });
}

use robopoker::cards::evaluator::Evaluator;
use robopoker::cards::hand::Hand;
use robopoker::cards::isomorphism::Isomorphism;
use robopoker::cards::observation::Observation;
use robopoker::cards::observations::ObservationIterator;
use robopoker::cards::street::Street;
use robopoker::cards::strength::Strength;
use robopoker::clustering::emd::EMD;
use robopoker::clustering::equity::Equity;
use robopoker::clustering::heuristic::Heuristic;
use robopoker::clustering::histogram::Histogram;
use robopoker::clustering::sinkhorn::Sinkhorn;
use robopoker::mccfr::rps::RPS;
use robopoker::mccfr::traits::Blueprint;
use robopoker::transport::coupling::Coupling;
use robopoker::Arbitrary;
