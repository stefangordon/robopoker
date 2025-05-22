use super::edge::Edge;
use super::game::Game;
use super::info::Info;
use super::turn::Turn;
use crate::cards::street::Street;
use std::io::Write;
use std::fs::File;
use std::mem::size_of;
use crate::Arbitrary;
use rustc_hash::FxHashMap;
use zstd::stream::{Encoder as ZEncoder, Decoder as ZDecoder};
use std::io::Seek;
use dashmap::DashMap;
use parking_lot::Mutex;
use crate::mccfr::types::policy::Policy;
use crate::mccfr::traits::info::Info as InfoTrait;
use smallvec::SmallVec;
use std::path::PathBuf;
use std::io::BufWriter;

pub struct Profile {
    pub(super) iterations: usize,
    pub(super) encounters: DashMap<Info, Mutex<FxHashMap<Edge, (crate::Probability, crate::Utility)>>>,
}

impl Default for Profile {
    fn default() -> Self {
        Self {
            iterations: 0,
            encounters: DashMap::new(),
        }
    }
}

impl Profile {
    fn name() -> String {
        "blueprint".to_string()
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
        self.encounters
            .get(info)
            .and_then(|bucket_mutex| {
                let guard = bucket_mutex.value().lock();
                guard.get(edge).map(|(w, _)| *w)
            })
            .unwrap_or_default()
    }
    fn sum_regret(&self, info: &Self::I, edge: &Self::E) -> crate::Utility {
        self.encounters
            .get(info)
            .and_then(|bucket_mutex| {
                let guard = bucket_mutex.value().lock();
                guard.get(edge).map(|(_, r)| *r)
            })
            .unwrap_or_default()
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
            let bucket = bucket_mutex.lock();
            for edge in choices_vec.iter().copied() {
                let regret_raw = bucket.get(&edge).map(|(_, r)| *r).unwrap_or_default();
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
        use crate::clustering::abstraction::Abstraction;
        use crate::gameplay::path::Path;
        use crate::mccfr::nlhe::info::Info;
        use byteorder::{ByteOrder, BE};
        use std::fs::File;
        use std::io::{BufReader, Read};

        const RECORD_SIZE: usize = 66; // 2-byte header already consumed + 64 payload

        let path = Self::path(Street::random());
        log::info!("{:<32}{:<32}", "loading     blueprint", path);

        let _file = File::open(&path).expect("open blueprint file");
        // Detect zstd magic (0x28 B5 2F FD)
        let mut magic = [0u8; 4];
        let mut file_for_magic = std::fs::File::open(&path).expect("open");
        let _ = file_for_magic.read_exact(&mut magic);
        file_for_magic.seek(std::io::SeekFrom::Start(0)).unwrap();
        let reader_inner: Box<dyn Read> = if magic == [0x28, 0xB5, 0x2F, 0xFD] {
            Box::new(ZDecoder::new(file_for_magic).expect("zstd decode"))
        } else {
            Box::new(file_for_magic)
        };

        let mut reader = BufReader::with_capacity(1024 * 1024, reader_inner);

        // Skip the 19-byte PG binary COPY header that precedes the first record
        {
            let mut skip = [0u8; 19];
            reader.read_exact(&mut skip).expect("skip pgcopy header");
        }

        let encounters: DashMap<Info, Mutex<FxHashMap<Edge, (crate::Probability, crate::Utility)>>> = DashMap::new();
        let mut header = [0u8; 2];
        let mut record = [0u8; RECORD_SIZE - 2]; // we already read the 2-byte header separately

        loop {
            if reader.read_exact(&mut header).is_err() {
                break; // EOF
            }
            let fields = u16::from_be_bytes(header);
            if fields == 0xFFFF {
                break; // trailer
            }
            debug_assert_eq!(fields, 6, "unexpected field count {}", fields);

            // Read remaining 64 bytes of the record in one shot
            reader.read_exact(&mut record).expect("record bytes");

            // Offsets inside `record`
            let mut offset = 0;
            // helper closures
            let mut read_len = |size: usize| {
                offset += 4; // skip length prefix (we trust it)
                let start = offset;
                let end = offset + size;
                offset = end;
                &record[start..end]
            };

            let history = Path::from(BE::read_u64(read_len(8)));
            let present = Abstraction::from(BE::read_u64(read_len(8)));
            let futures = Path::from(BE::read_u64(read_len(8)));
            let edge = Edge::from(BE::read_u64(read_len(8)));
            let regret = BE::read_f32(read_len(4));
            let policy = BE::read_f32(read_len(4));

            let bucket = Info::from((history, present, futures));
            encounters
                .entry(bucket)
                .or_insert_with(|| Mutex::new(FxHashMap::default()))
                .lock()
                .entry(edge)
                .or_insert((policy, regret));
        }

        Self {
            encounters,
            iterations: 0,
        }
    }
    fn save(&self) {
        const N_FIELDS: u16 = 6;
        let path_str = Self::path(Street::random());
        let path = std::path::Path::new(&path_str);

        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("Failed to create parent directories");
        }

        let file = File::create(path).expect(&format!("Failed to create file at {}", path_str));
        let buf_writer = BufWriter::with_capacity(8 * 1024 * 1024, file);
        let mut writer = ZEncoder::new(buf_writer, 0).expect("zstd encoder");
        writer.multithread(2 as u32).expect("failed to set multithreading");
        let total_buckets = self.encounters.len();
        log::info!("Saving blueprint to {} ({} info-sets)", path_str, total_buckets);

        // For progress tracking
        let mut processed_buckets = 0;
        let buckets_one_percent = total_buckets / 100;
        // Ensure we'll log at least some progress messages
        let log_every_n = if buckets_one_percent > 0 { buckets_one_percent } else { 1 };

        use crate::Arbitrary;
        use byteorder::WriteBytesExt;
        use byteorder::BE;
        log::info!("{:<32}{:<32}", "saving      blueprint", path_str);
        writer.write_all(Self::header()).expect("header");
        for bucket_entry in self.encounters.iter() {
            let bucket = bucket_entry.key();
            let edges = bucket_entry.value().lock();
            for (edge, memory) in edges.iter() {
                writer.write_u16::<BE>(N_FIELDS).unwrap();
                writer.write_u32::<BE>(size_of::<u64>() as u32).unwrap();
                writer.write_u64::<BE>(u64::from(*bucket.history())).unwrap();
                writer.write_u32::<BE>(size_of::<u64>() as u32).unwrap();
                writer.write_u64::<BE>(u64::from(*bucket.present())).unwrap();
                writer.write_u32::<BE>(size_of::<u64>() as u32).unwrap();
                writer.write_u64::<BE>(u64::from(*bucket.futures())).unwrap();
                writer.write_u32::<BE>(size_of::<u64>() as u32).unwrap();
                writer.write_u64::<BE>(u64::from(edge.clone())).unwrap();
                writer.write_u32::<BE>(size_of::<f32>() as u32).unwrap();
                writer.write_f32::<BE>(memory.1).unwrap();
                writer.write_u32::<BE>(size_of::<f32>() as u32).unwrap();
                writer.write_f32::<BE>(memory.0).unwrap();
            }

            // Track progress by buckets, not individual edges
            processed_buckets += 1;
            if processed_buckets % log_every_n == 0 || processed_buckets == total_buckets {
                let percentage = (processed_buckets * 100) / total_buckets;
                log::info!("Saving blueprint progress: {}%", percentage);
            }
        }
        writer.write_u16::<BE>(Self::footer()).expect("trailer");
        writer.finish().expect("finish zstd");
    }
}

impl Profile {
    /// Populates the profile with synthetic data for benchmarking purposes.
    ///
    /// `bucket_count`: number of distinct information sets to generate.
    /// `edges_per_bucket`: number of edges per information set.
    pub fn populate_dummy(&mut self, bucket_count: usize, edges_per_bucket: usize) {
        use rand::Rng;
        use rustc_hash::FxHashMap;

        for _ in 0..bucket_count {
            let info = Info::random();
            let bucket_mutex = self
                .encounters
                .entry(info)
                .or_insert_with(|| Mutex::new(FxHashMap::with_capacity_and_hasher(4, Default::default())));
            let mut bucket = bucket_mutex.lock();
            for _ in 0..edges_per_bucket {
                let edge = Edge::random();
                let policy: f32 = rand::thread_rng().gen_range(0.0..1.0);
                let regret: f32 = rand::thread_rng().gen_range(-1.0..1.0);
                bucket.insert(edge, (policy, regret));
            }
        }

        // Update iterations counter to reflect synthetic data
        self.iterations += bucket_count;
    }
}