use super::edge::Edge;
use super::game::Game;
use super::info::Info;
use super::turn::Turn;
use crate::cards::street::Street;
use crate::Arbitrary;
use zstd::stream::Decoder as ZDecoder;
use dashmap::DashMap;
use parking_lot::RwLock;
use crate::mccfr::types::policy::Policy;
use crate::mccfr::types::decision::Decision;
use crate::mccfr::traits::info::Info as InfoTrait;
use smallvec::SmallVec;
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

// Using SmallVec (inline capacity 4) instead of a per-infoset HashMap dramatically
// reduces overhead. The majority of infosets have ≤4 edges.
type Bucket = SmallVec<[(Edge, (f16, crate::Utility)); 4]>;

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

    /// Log aggregate statistics for regret and policy vectors instead of writing to disk.
    /// Enabled by setting environment variable `BLUEPRINT_STATS=1` before running.
    #[cfg(feature = "native")]
    pub fn log_stats(&self) {
        use std::f32::{INFINITY, NEG_INFINITY};

        let mut infosets = 0usize;
        let mut total_edges = 0usize;
        let mut min_edges = usize::MAX;
        let mut max_edges = 0usize;

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

            for (_, (policy, regret)) in bucket.iter() {
                let pol_f32 = f32::from(*policy);
                policy_min = policy_min.min(pol_f32);
                policy_max = policy_max.max(pol_f32);
                regret_min = regret_min.min(*regret);
                regret_max = regret_max.max(*regret);

                // Bucket regrets
                if *regret == 0.0 {
                    zero_regret_count += 1;
                } else if *regret < -1_000_000.0 {
                    regret_buckets[0] += 1;
                } else if *regret < -10_000.0 {
                    regret_buckets[1] += 1;
                } else if *regret < -100.0 {
                    regret_buckets[2] += 1;
                } else if *regret < 0.0 {
                    regret_buckets[3] += 1;
                } else if *regret < 100.0 {
                    regret_buckets[4] += 1;
                } else if *regret < 10_000.0 {
                    regret_buckets[5] += 1;
                } else if *regret < 1_000_000.0 {
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
            .or_insert_with(|| RwLock::new(SmallVec::new()));
        let mut bucket = bucket_mutex.value().write();
        // Normalise weights to sum to 1.
        let total: f32 = decisions.iter().map(|d| d.weight()).sum();
        let total = total.max(crate::POLICY_MIN);
        for d in decisions {
            let w = d.weight() / total;
            bucket.push((d.edge(), (half::f16::from_f32(w), 0.0)));
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
                .or_insert_with(|| RwLock::new(SmallVec::new()));
            let mut bucket = bucket_mutex.value().write();
            // Update regrets
            for (edge, delta) in regret_vec {
                if let Some(existing) = bucket.iter_mut().find(|(e, _)| *e == edge) {
                    let new_regret = existing.1 .1 + delta;
                    existing.1 .1 = self::safe_clamp(new_regret, current_min, current_max);
                } else {
                    // Check delta for NaN before storing
                    let safe_delta = self::safe_clamp(delta, current_min, current_max);
                    bucket.push((edge, (f16::from_f32(0.0), safe_delta)));
                }
            }
            // Update policies
            for (edge, delta) in policy_vec {
                if let Some(existing) = bucket.iter_mut().find(|(e, _)| *e == edge) {
                    let prev = f32::from(existing.1 .0);
                    let new_policy = prev + delta;

                    // Fast check for invalid values
                    if new_policy.is_nan() || new_policy < 0.0 {
                        existing.1 .0 = f16::from_f32(crate::POLICY_MIN);
                    } else {
                        existing.1 .0 = f16::from_f32(new_policy);
                    }
                } else {
                    // Check delta for validity before storing
                    let safe_delta = if delta.is_nan() || delta < 0.0 { crate::POLICY_MIN } else { delta };
                    bucket.push((edge, (f16::from_f32(safe_delta), 0.0)));
                }
            }
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
    fn sum_policy(&self, info: &Self::I, edge: &Self::E) -> crate::Probability {
        if let Some(bucket_mutex) = self.encounters.get(info) {
            let bucket = bucket_mutex.value().read();
            bucket
                .iter()
                .find_map(|(e, (p, _))| if e == edge { Some(f32::from(*p)) } else { None })
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
                .find_map(|(e, (_, r))| if e == edge { Some(*r) } else { None })
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

    // -------------------------------------------------------------------
    // Optimized policy_vector: cache bucket map once per infoset to avoid
    // repeated DashMap lookups and mutex locking for each edge.
    // -------------------------------------------------------------------
    fn policy_vector(
        &self,
        infoset: &crate::mccfr::structs::infoset::InfoSet<Self::T, Self::E, Self::G, Self::I>,
    ) -> crate::mccfr::types::policy::Policy<Self::E> {
        let info = infoset.info();

        let choices_vec = info.choices();
        let mut small: SmallVec<[(Self::E, f32); 8]> = SmallVec::with_capacity(choices_vec.len().min(8));

        if let Some(bucket_mutex) = self.encounters.get(&info) {
            let bucket = bucket_mutex.value().read();
            for edge in choices_vec.iter().copied() {
                let regret_raw = bucket
                    .iter()
                    .find_map(|(e, (_, r))| if *e == edge { Some(*r) } else { None })
                    .unwrap_or_default();
                small.push((edge, regret_raw.max(crate::POLICY_MIN)));
            }
        } else {
            for edge in choices_vec.iter().copied() {
                small.push((edge, crate::POLICY_MIN));
            }
        }

        let mut regrets_vec: Policy<Self::E> = small.into_vec();
        let denominator: crate::Utility = regrets_vec.iter().map(|(_, r)| r).sum();
        regrets_vec
            .iter_mut()
            .for_each(|(_, r)| *r /= denominator);
        regrets_vec
            .into_iter()
            .collect()
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
                .or_insert_with(|| RwLock::new(SmallVec::new()));
            let mut bucket = bucket_mutex.value().write();
            for _ in 0..edges_per_bucket {
                let edge = Edge::random();
                let policy: f32 = rand::thread_rng().gen_range(0.0..1.0);
                let regret: f32 = rand::thread_rng().gen_range(-1.0..1.0);
                bucket.push((edge, (f16::from_f32(policy), regret)));
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

        // Calculate total records
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

        // Define encodings
        let encodings = vec![
            vec![Encoding::Plain],      // history
            vec![Encoding::Plain],      // present
            vec![Encoding::Plain],      // futures
            vec![Encoding::Plain],      // edge
            vec![Encoding::Plain],      // regret
            vec![Encoding::Plain],      // policy
        ];

        // Write options - optimized for speed
        let options = WriteOptions {
            write_statistics: true, // Keep statistics for better query performance
            compression: CompressionOptions::Zstd(Some(arrow2::io::parquet::write::ZstdLevel::try_new(1).unwrap())), // Fast ZSTD compression level 1
            version: Version::V2,
            data_pagesize_limit: Some(1024 * 1024), // Larger 1MB pages for better throughput
        };

        // Create output file and writer
        let file = File::create(path).expect(&format!("Failed to create file at {}", path_str));
        let writer_inner = BufWriter::with_capacity(8 * 1024 * 1024, file); // 8MB buffer
        let mut writer = FileWriter::try_new(writer_inner, schema.clone(), options)
            .expect("Failed to create parquet writer");

        // Stream data in chunks to avoid doubling memory usage
        const RECORDS_PER_CHUNK: usize = 1_000_000; // Process 1M records at a time
        let mut _written_records = 0;

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
                chunk_histories.push(u64::from(*bucket.history()) as i64);
                chunk_presents.push(u64::from(*bucket.present()) as i64);
                chunk_futures.push(u64::from(*bucket.futures()) as i64);
                chunk_edges.push(u64::from(edge.clone()) as u32); // Note: Edge values must fit in u32
                chunk_regrets.push(*regret);
                chunk_policies.push(f32::from(*policy));

                _written_records += 1;
                progress.inc(1);

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

                    // Clear chunks for next batch
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

        let mut _total_records = 0;

        // Process each row group
        for (_rg_idx, row_group) in metadata.row_groups.iter().enumerate() {
            // Create a file reader for this row group
            let chunks = FileReader::new(
                file.try_clone().expect("Failed to clone file handle"),
                vec![row_group.clone()],
                schema.clone(),
                None,
                None,
                None,
            );

            // Read the chunk
            for chunk_result in chunks {
                let chunk = chunk_result.expect("Failed to read chunk");

                // Get arrays from chunk
                let arrays = chunk.arrays();
                if arrays.len() != 6 {
                    panic!("Expected 6 columns, got {}", arrays.len());
                }

                let histories = arrays[0].as_any().downcast_ref::<Int64Array>()
                    .expect("Failed to cast history column");
                let presents = arrays[1].as_any().downcast_ref::<Int64Array>()
                    .expect("Failed to cast present column");
                let futures = arrays[2].as_any().downcast_ref::<Int64Array>()
                    .expect("Failed to cast futures column");
                let edges = arrays[3].as_any().downcast_ref::<UInt32Array>()
                    .expect("Failed to cast edge column");
                let regrets = arrays[4].as_any().downcast_ref::<Float32Array>()
                    .expect("Failed to cast regret column");
                let policies = arrays[5].as_any().downcast_ref::<Float32Array>()
                    .expect("Failed to cast policy column");

                let num_rows = chunk.len();

                // Process records
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
                        .or_insert_with(|| RwLock::new(SmallVec::new()));
                    let mut bucket = bucket_mutex.value().write();
                    bucket.push((edge, (f16::from_f32(policy), regret)));

                    _total_records += 1;
                    progress.inc(1);
                }
            }
        }

        progress.finish_with_message("Blueprint loaded from parquet");

        Self {
            encounters,
            iterations: 0,
            regret_min: crate::REGRET_MIN,
            regret_max: crate::REGRET_MAX,
        }
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
                .or_insert_with(|| RwLock::new(SmallVec::new()));
            let mut bucket = bucket_mutex.value().write();
            bucket.push((edge, (f16::from_f32(policy), regret)));
        }

        Self {
            encounters,
            iterations: 0,
            regret_min: crate::REGRET_MIN,
            regret_max: crate::REGRET_MAX,
        }
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
        let mut warm_start = self.warm_start;
        // Populate profile with warm-start decisions if any.
        (profile, warm_start)
    }
}