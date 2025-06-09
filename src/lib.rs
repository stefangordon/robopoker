#[cfg(all(feature = "native", feature = "subgame"))]
pub mod analysis;
#[cfg(feature = "native")]
pub mod modeldata;
#[cfg(feature = "native")]
pub mod players;
#[cfg(feature = "native")]
pub mod save;

pub mod cards;
pub mod clustering;
pub mod gameplay;
pub mod mccfr;
pub mod search;
pub mod transport;
pub mod wasm;

#[cfg(feature = "subgame")]
pub use analysis::Server;

pub use cards::*;
pub use gameplay::*;
pub use mccfr::*;
pub use transport::*;
pub use wasm::*;

#[cfg(feature = "native")]
static INTERRUPTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// dimensional analysis types
type Chips = i16;
type Equity = f32;
type Energy = f32;
type Entropy = f32;
type Utility = f32;
type Probability = f32;

// game tree parameters
const N: usize = 2; // from 2
const STACK: Chips = 400;
const B_BLIND: Chips = 2;
const S_BLIND: Chips = 1;
const MAX_RAISE_REPEATS: usize = 3;
const MAX_DEPTH_SUBGAME: usize = 16;

/// sinkhorn optimal transport parameters
const SINKHORN_TEMPERATURE: Entropy = 0.025;
const SINKHORN_ITERATIONS: usize = 128;
const SINKHORN_TOLERANCE: Energy = 0.001;

// kmeans clustering parameters
#[cfg(not(feature = "shortdeck"))]
mod kmeans_constants {
    pub const KMEANS_FLOP_TRAINING_ITERATIONS: usize = 20;
    pub const KMEANS_TURN_TRAINING_ITERATIONS: usize = 24;
    pub const KMEANS_FLOP_CLUSTER_COUNT: usize = 128;
    pub const KMEANS_TURN_CLUSTER_COUNT: usize = 144;
    pub const KMEANS_EQTY_CLUSTER_COUNT: usize = 101;
}

#[cfg(feature = "shortdeck")]
mod kmeans_constants {
    pub const KMEANS_FLOP_TRAINING_ITERATIONS: usize = 15;
    pub const KMEANS_TURN_TRAINING_ITERATIONS: usize = 18;
    pub const KMEANS_FLOP_CLUSTER_COUNT: usize = 96;
    pub const KMEANS_TURN_CLUSTER_COUNT: usize = 96;
    pub const KMEANS_EQTY_CLUSTER_COUNT: usize = 96;
}

use kmeans_constants::*;

/// rps mccfr parameteres
const ASYMMETRIC_UTILITY: f32 = 2.0;
const CFR_BATCH_SIZE_RPS: usize = 1;
const CFR_TREE_COUNT_RPS: usize = 8192;

/// checkpointing interval (in hours) for periodic profile saves during training
#[cfg(feature = "native")]
pub const CHECKPOINT_HOURS: u64 = 8;

// nlhe mccfr parameters
const CFR_BATCH_SIZE_NLHE: usize = 1024;
const CFR_TREE_COUNT_NLHE: usize = 0x2FAF080;
const RBP_WARMUP_TRAVERSALS: u64 = 0x1000000; // 16.7M traversals warmup before RBP starts

/// profile average sampling parameters
const SAMPLING_THRESHOLD: Entropy = 1.0;
const SAMPLING_ACTIVATION: Energy = 0.2;
const SAMPLING_EXPLORATION: Probability = 0.01;

// regret matching parameters with clamping to prevent numerical instability
const POLICY_MIN: Probability = Probability::MIN_POSITIVE;
const REGRET_MIN: Utility = -500_000_000.0;
const REGRET_MAX: Utility = 500_000_000.0;

#[derive(clap::Parser, Debug)]
#[command(author, version, about, long_about = None)]
#[cfg(feature = "native")]
/// Simple program to run robopoker routines
pub struct Args {
    /// Run the clustering algorithm
    #[arg(long)]
    pub cluster: bool,
    /// Run the MCCFR training
    #[arg(long)]
    pub trainer: bool,
    /// Publish results to the database
    #[arg(long)]
    pub publish: bool,
    /// Run the analysis server
    #[arg(long)]
    #[cfg(feature = "subgame")]
    pub analyze: bool,
    /// Run exploitability evaluation
    #[arg(long)]
    pub evaluate: bool,
    /// Convert blueprint to PostgreSQL-compatible format
    #[arg(long)]
    pub convert_blueprint_to_postgres: bool,
    /// Convert blueprint to Parquet format
    #[arg(long)]
    pub convert_blueprint_to_parquet: bool,
    /// Generate model training data in Parquet format
    #[arg(long)]
    pub genmodeldata: bool,
}

/// trait for random generation, mainly (strictly?) for testing
pub trait Arbitrary {
    fn random() -> Self;
}

/// progress bar
#[cfg(feature = "native")]
pub fn progress(n: usize) -> indicatif::ProgressBar {
    let tick = std::time::Duration::from_secs(60);
    // Show items per second throughput in addition to elapsed time, percent and bar.
    // The long tick interval (60s) limits redraw frequency to reduce overhead.
    let style =
        "{spinner:.cyan} [{pos}/{len}] {elapsed} @ {per_sec:>12} ~ {percent:>3}% {wide_bar:.cyan}";
    let style = indicatif::ProgressStyle::with_template(style).unwrap();
    let progress = indicatif::ProgressBar::new(n as u64);
    progress.set_style(style);
    progress.enable_steady_tick(tick);
    progress
}

/// initialize logging and setup graceful interrupt listener
#[cfg(feature = "native")]
pub fn logs() {
    std::fs::create_dir_all("logs").expect("create logs directory");
    let config = simplelog::ConfigBuilder::new()
        .set_location_level(log::LevelFilter::Off)
        .set_target_level(log::LevelFilter::Off)
        .set_thread_level(log::LevelFilter::Off)
        .build();
    let time = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("time moves slow")
        .as_secs();
    let file = simplelog::WriteLogger::new(
        log::LevelFilter::Debug,
        config.clone(),
        std::fs::File::create(format!("logs/{}.log", time)).expect("create log file"),
    );
    let term = simplelog::TermLogger::new(
        log::LevelFilter::Info,
        config.clone(),
        simplelog::TerminalMode::Mixed,
        simplelog::ColorChoice::Auto,
    );
    simplelog::CombinedLogger::init(vec![term, file]).expect("initialize logger");
}

/// get a database connection and return the client
#[cfg(feature = "native")]
pub async fn db() -> std::sync::Arc<tokio_postgres::Client> {
    log::info!("connecting to database");
    let tls = tokio_postgres::tls::NoTls;
    let ref url = std::env::var("DB_URL").expect("DB_URL must be set");
    let (client, connection) = tokio_postgres::connect(url, tls)
        .await
        .expect("database connection failed");
    tokio::spawn(connection);
    std::sync::Arc::new(client)
}

#[cfg(feature = "native")]
/// keyboard interruption for training
/// spawn a thread to listen for 'q' input to gracefully interrupt training
pub fn interrupts() {
    // handle ctrl+c for immediate exit
    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.unwrap();
        println!();
        log::warn!("Ctrl+C received, exiting immediately");
        std::process::exit(0);
    });
    // handle 'q' input for graceful interrupt
    std::thread::spawn(|| {
        log::info!("training started. type 'Q + Enter' to gracefully interrupt.");
        let ref mut buffer = String::new();
        loop {
            buffer.clear();
            if let Ok(_) = std::io::stdin().read_line(buffer) {
                if buffer.trim().to_uppercase() == "Q" {
                    log::warn!("graceful interrupt requested, finishing current batch...");
                    INTERRUPTED.store(true, std::sync::atomic::Ordering::Relaxed);
                    break;
                }
            }
        }
    });
}
