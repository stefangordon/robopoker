use super::edge::Edge;
use super::game::Game;
use super::info::Info;
use super::turn::Turn;
use crate::cards::street::Street;
use crate::Arbitrary;
use zstd::stream::Decoder as ZDecoder;
use dashmap::DashMap;
use parking_lot::Mutex;
use crate::mccfr::types::policy::Policy;
use crate::mccfr::traits::info::Info as InfoTrait;
use smallvec::SmallVec;
use std::path::PathBuf;
use ahash::RandomState;
use half::f16;
#[cfg(feature = "native")]
use memmap2::MmapOptions;

// File format constants and specifications
//
// Two formats are supported:
// 1. Legacy: zstd-compressed PostgreSQL binary COPY format (detected by zstd magic bytes)
// 2. Current: Uncompressed memory-mapped binary format (default for new saves)
//
// Legacy Format: [zstd header][19-byte PG header][records...][trailer]
// Current Format: [8-byte record count][40-byte records...]
//
// Current format record layout (little-endian):
// - history: 8 bytes (u64)
// - present: 8 bytes (u64)
// - futures: 8 bytes (u64)
// - edge: 8 bytes (u64)
// - regret: 4 bytes (f32)
// - policy: 4 bytes (f32)

const ZSTD_MAGIC: [u8; 4] = [0x28, 0xB5, 0x2F, 0xFD];
const MMAP_RECORD_SIZE: usize = 40; // 8+8+8+8+4+4 bytes per record
const MMAP_HEADER_SIZE: usize = 8; // record count header
const PGCOPY_RECORD_SIZE: usize = 66; // 2-byte header already consumed + 64 payload
const PGCOPY_HEADER_SIZE: usize = 19; // PostgreSQL binary COPY header
const CHUNK_SIZE: usize = 1000; // Processing chunk size for better cache performance

// Using SmallVec (inline capacity 4) instead of a per-infoset HashMap dramatically
// reduces overhead. The majority of infosets have ≤4 edges.
type Bucket = SmallVec<[(Edge, (f16, crate::Utility)); 4]>;

pub struct Profile {
    pub(super) iterations: usize,
    pub(super) encounters: DashMap<Info, Mutex<Bucket>, RandomState>,
}

impl Default for Profile {
    fn default() -> Self {
        Self {
            iterations: 0,
            encounters: DashMap::with_hasher(RandomState::default()),
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
            let bucket = bucket_entry.value().lock();
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
            let bucket = bucket_mutex.value().lock();
            bucket
                .iter()
                .find_map(|(e, (p, _))| if e == edge { Some(f32::from(*p)) } else { None })
                .unwrap_or_default()
        } else {
            0.0
        }
    }
    fn sum_regret(&self, info: &Self::I, edge: &Self::E) -> crate::Utility {
        if let Some(bucket_mutex) = self.encounters.get(info) {
            let bucket = bucket_mutex.value().lock();
            bucket
                .iter()
                .find_map(|(e, (_, r))| if e == edge { Some(*r) } else { None })
                .unwrap_or_default()
        } else {
            0.0
        }
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
            let bucket = bucket_mutex.value().lock();
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
        );
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
            log::debug!("Detected new mmap binary format");
            Self::load_mmap()
        }
    }
        fn save(&self) {
        if std::env::var("BLUEPRINT_STATS").is_ok() {
            // If caller only wants statistics, skip expensive serialization
            self.log_stats();
        } else {
            // Use optimized mmap format for all saves
            self.save_mmap();
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
                .or_insert_with(|| Mutex::new(SmallVec::new()));
            let mut bucket = bucket_mutex.lock();
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

        /// Save profile data using memory-mapped I/O with the new binary format.
    /// This is now the default save format. Maps a file into memory and writes directly to the mapped region.
    #[cfg(feature = "native")]
    pub fn save_mmap(&self) {
        use byteorder::{ByteOrder, LittleEndian};
        use crate::save::disk::Disk;
        use std::fs::OpenOptions;

        let path_str = Self::path(Street::random());
        let path = std::path::Path::new(&path_str);

        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("Failed to create parent directories");
        }

        // Calculate total records and file size
        let total_records: usize = self.encounters.iter()
            .map(|entry| entry.value().lock().len())
            .sum();
        let file_size = MMAP_HEADER_SIZE + (total_records * MMAP_RECORD_SIZE);

        log::info!("Saving blueprint to {} ({} records, {} bytes)",
                   path_str, total_records, file_size);

        // Create and size the file
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(true)
            .open(path)
            .expect(&format!("Failed to create file at {}", path_str));

        file.set_len(file_size as u64).expect("Failed to set file size");

        // Memory map the file for writing
        let mut mmap = unsafe {
            MmapOptions::new()
                .map_mut(&file)
                .expect("Failed to mmap file for writing")
        };

        // Write header: total record count
        LittleEndian::write_u64(&mut mmap[0..MMAP_HEADER_SIZE], total_records as u64);

        // Write records directly to mapped memory
        let mut offset = MMAP_HEADER_SIZE;
        let mut written_records = 0;
        let log_every_n = (total_records / 100).max(1);

        for bucket_entry in self.encounters.iter() {
            let bucket = bucket_entry.key();
            let edges = bucket_entry.value().lock();

            for (edge, (policy, regret)) in edges.iter() {
                // Write record: history(8) + present(8) + futures(8) + edge(8) + regret(4) + policy(4)
                LittleEndian::write_u64(&mut mmap[offset..offset+8], u64::from(*bucket.history()));
                LittleEndian::write_u64(&mut mmap[offset+8..offset+16], u64::from(*bucket.present()));
                LittleEndian::write_u64(&mut mmap[offset+16..offset+24], u64::from(*bucket.futures()));
                LittleEndian::write_u64(&mut mmap[offset+24..offset+32], u64::from(edge.clone()));
                LittleEndian::write_f32(&mut mmap[offset+32..offset+36], *regret);
                LittleEndian::write_f32(&mut mmap[offset+36..offset+40], f32::from(*policy));

                offset += MMAP_RECORD_SIZE;
                written_records += 1;

                if written_records % log_every_n == 0 || written_records == total_records {
                    let percentage = (written_records * 100) / total_records;
                    log::info!("Saving blueprint progress: {}%", percentage);
                }
            }
        }

        // Ensure data is flushed to disk
        mmap.flush().expect("Failed to flush mmap");
        log::info!("Completed saving {} records to {}", written_records, path_str);
    }

        /// Load profile data using memory-mapped I/O.
    /// Maps the file into memory and reads directly from the mapped region.
    #[cfg(feature = "native")]
    pub fn load_mmap() -> Self {
        use byteorder::{ByteOrder, LittleEndian};
        use crate::clustering::abstraction::Abstraction;
        use crate::gameplay::path::Path;
        use std::fs::File;
        use crate::save::disk::Disk;

        let path_str = Self::path(Street::random());
        log::info!("{:<32}{:<32}", "loading     blueprint", path_str);

        let file = File::open(&path_str).expect(&format!("Failed to open blueprint file: {}", path_str));
        let mmap = unsafe { MmapOptions::new().map(&file).expect("Failed to mmap file") };

        // Validate file size and read header
        if mmap.len() < MMAP_HEADER_SIZE {
            panic!("File too small to contain header: {} bytes", mmap.len());
        }
        let total_records = LittleEndian::read_u64(&mmap[0..MMAP_HEADER_SIZE]) as usize;

        let expected_size = MMAP_HEADER_SIZE + (total_records * MMAP_RECORD_SIZE);
        if mmap.len() < expected_size {
            panic!("File size mismatch: expected {} bytes, got {}", expected_size, mmap.len());
        }

        log::info!("Loading {} records from memory-mapped file", total_records);

        // Pre-allocate with estimated capacity for better performance
        let encounters: DashMap<Info, Mutex<Bucket>, RandomState> =
            DashMap::with_capacity_and_hasher(total_records / 4, RandomState::default());

        let log_every_n = (total_records / 100).max(1);
        let mut records_processed = 0;

        // Process records in chunks for better cache performance
        for chunk_start in (0..total_records).step_by(CHUNK_SIZE) {
            let chunk_end = (chunk_start + CHUNK_SIZE).min(total_records);

            for i in chunk_start..chunk_end {
                let record_offset = MMAP_HEADER_SIZE + (i * MMAP_RECORD_SIZE);

                // Read record directly from mapped memory: history(8) + present(8) + futures(8) + edge(8) + regret(4) + policy(4)
                let history = Path::from(LittleEndian::read_u64(&mmap[record_offset..record_offset+8]));
                let present = Abstraction::from(LittleEndian::read_u64(&mmap[record_offset+8..record_offset+16]));
                let futures = Path::from(LittleEndian::read_u64(&mmap[record_offset+16..record_offset+24]));
                let edge = Edge::from(LittleEndian::read_u64(&mmap[record_offset+24..record_offset+32]));
                let regret = LittleEndian::read_f32(&mmap[record_offset+32..record_offset+36]);
                let policy = LittleEndian::read_f32(&mmap[record_offset+36..record_offset+40]);

                let info = Info::from((history, present, futures));
                let bucket_mutex = encounters
                    .entry(info)
                    .or_insert_with(|| Mutex::new(SmallVec::new()));
                let mut bucket = bucket_mutex.lock();
                bucket.push((edge, (f16::from_f32(policy), regret)));

                records_processed += 1;
            }

            if records_processed % log_every_n == 0 || records_processed == total_records {
                let percentage = (records_processed * 100) / total_records;
                log::info!("Loading blueprint progress: {}%", percentage);
            }
        }

        log::info!("Loaded {} records from memory-mapped file", total_records);

        Self {
            encounters,
            iterations: 0,
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

        let encounters: DashMap<Info, Mutex<Bucket>, RandomState> = DashMap::with_hasher(RandomState::default());
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
            let regret = BE::read_f32(read_field(4));
            let policy = BE::read_f32(read_field(4));

            let info = Info::from((history, present, futures));
            let bucket_mutex = encounters
                .entry(info)
                .or_insert_with(|| Mutex::new(SmallVec::new()));
            let mut bucket = bucket_mutex.lock();
            bucket.push((edge, (f16::from_f32(policy), regret)));
        }

        Self {
            encounters,
            iterations: 0,
        }
    }
}