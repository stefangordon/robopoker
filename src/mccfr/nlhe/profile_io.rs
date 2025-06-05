use super::profile::{Profile, FxBuildHasher};
use super::compact_bucket::CompactBucket;
use super::edge::Edge;
use super::info::Info;
use crate::cards::street::Street;
use crate::clustering::abstraction::Abstraction;
use crate::gameplay::path::Path;
use crate::Arbitrary;

#[cfg(feature = "native")]
use arrow2::{
    array::{Array, Float32Array, Int64Array, UInt32Array},
    chunk::Chunk,
    datatypes::{DataType, Field, Schema},
    io::parquet::{
        read::{infer_schema, read_metadata, FileReader},
        write::{
            CompressionOptions, Encoding, FileWriter, RowGroupIterator, Version, WriteOptions,
        },
    },
};
use dashmap::DashMap;
use half::f16;
use parking_lot::RwLock;
use std::path::PathBuf;
use std::sync::Arc;
use zstd::stream::Decoder as ZDecoder;

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

impl Profile {
    /// Log aggregate statistics for regret and policy vectors instead of writing to disk.
    /// Enabled by setting environment variable `BLUEPRINT_STATS=1` before running.
    #[cfg(feature = "native")]
    pub fn log_stats(&self) {
        use std::f32::{INFINITY, NEG_INFINITY};

        // Hard cap at 100 million records to prevent excessive processing time
        const MAX_RECORDS: usize = 100_000_000;

        // Structure to hold stats
        struct Stats {
            infosets: usize,
            total_edges: usize,
            min_edges: usize,
            max_edges: usize,
            edge_count_dist: [usize; 14],
            policy_min: f32,
            policy_max: f32,
            regret_min: f32,
            regret_max: f32,
            regret_buckets: [usize; 8],
            policy_buckets: [usize; 5],
            zero_regret_count: usize,
            // Convergence metrics fields
            near_zero_regrets: usize,
            high_confidence_actions: usize,
            total_entropy: f64,
        }

        // Constants for convergence metrics
        const ZERO_THRESHOLD: f32 = 1.0;
        const HIGH_CONFIDENCE_THRESHOLD: f32 = 0.9;

        let total_infosets = self.encounters.len();
        let process_limit = total_infosets.min(MAX_RECORDS);

        log::info!(
            "Blueprint has {} infosets, processing first {} ({:.1}%)",
            total_infosets,
            process_limit,
            (process_limit as f64 / total_infosets as f64) * 100.0
        );

        // Initialize stats
        let mut stats = Stats {
            infosets: 0,
            total_edges: 0,
            min_edges: usize::MAX,
            max_edges: 0,
            edge_count_dist: [0; 14],
            policy_min: INFINITY,
            policy_max: NEG_INFINITY,
            regret_min: INFINITY,
            regret_max: NEG_INFINITY,
            regret_buckets: [0; 8],
            policy_buckets: [0; 5],
            zero_regret_count: 0,
            near_zero_regrets: 0,
            high_confidence_actions: 0,
            total_entropy: 0.0,
        };

        // Process records sequentially up to the limit
        let mut processed = 0;
        for bucket_entry in self.encounters.iter() {
            if processed >= process_limit {
                break;
            }

            processed += 1;
            stats.infosets += 1;

            let bucket = bucket_entry.value().read();
            let count = bucket.len();
            stats.total_edges += count;
            stats.min_edges = stats.min_edges.min(count);
            stats.max_edges = stats.max_edges.max(count);

            // Track edge count distribution
            if count < stats.edge_count_dist.len() {
                stats.edge_count_dist[count] += 1;
            }

            // For convergence metrics calculation
            let mut weights: Vec<f32> = Vec::with_capacity(count);

            for (_, (policy, regret)) in bucket.iter() {
                let pol_f32 = f32::from(policy);
                stats.policy_min = stats.policy_min.min(pol_f32);
                stats.policy_max = stats.policy_max.max(pol_f32);
                stats.regret_min = stats.regret_min.min(regret);
                stats.regret_max = stats.regret_max.max(regret);

                // Count near-zero regrets
                if regret.abs() < ZERO_THRESHOLD {
                    stats.near_zero_regrets += 1;
                }

                // Bucket regrets
                if regret == 0.0 {
                    stats.zero_regret_count += 1;
                } else if regret < -1_000_000.0 {
                    stats.regret_buckets[0] += 1;
                } else if regret < -10_000.0 {
                    stats.regret_buckets[1] += 1;
                } else if regret < -100.0 {
                    stats.regret_buckets[2] += 1;
                } else if regret < 0.0 {
                    stats.regret_buckets[3] += 1;
                } else if regret < 100.0 {
                    stats.regret_buckets[4] += 1;
                } else if regret < 10_000.0 {
                    stats.regret_buckets[5] += 1;
                } else if regret < 1_000_000.0 {
                    stats.regret_buckets[6] += 1;
                } else {
                    stats.regret_buckets[7] += 1;
                }

                // Bucket policies
                if pol_f32 < 1.0 {
                    stats.policy_buckets[0] += 1;
                } else if pol_f32 < 10.0 {
                    stats.policy_buckets[1] += 1;
                } else if pol_f32 < 100.0 {
                    stats.policy_buckets[2] += 1;
                } else if pol_f32 < 1000.0 {
                    stats.policy_buckets[3] += 1;
                } else {
                    stats.policy_buckets[4] += 1;
                }

                // Collect for entropy calculation
                weights.push(pol_f32.max(0.0));
            }

            // Calculate entropy for this infoset
            let sum: f32 = weights.iter().sum();
            if sum > 0.0 {
                let mut entropy = 0.0f64;
                let mut max_prob = 0.0f32;

                for w in &weights {
                    let p = w / sum;
                    if p > max_prob {
                        max_prob = p;
                    }
                    if p > 0.0 {
                        entropy -= (p as f64) * (p as f64).ln();
                    }
                }

                if max_prob > HIGH_CONFIDENCE_THRESHOLD {
                    stats.high_confidence_actions += 1;
                }

                stats.total_entropy += entropy;
            }

            // Log progress every 10M records
            if processed % 10_000_000 == 0 {
                log::info!("Processed {} / {} infosets", processed, process_limit);
            }
        }

        // Fix min_edges if no records were processed
        if stats.min_edges == usize::MAX {
            stats.min_edges = 0;
        }

        let avg_edges = if stats.infosets > 0 {
            stats.total_edges as f64 / stats.infosets as f64
        } else {
            0.0
        };

        // Calculate convergence metrics
        let convergence_ratio = if stats.total_edges > 0 {
            stats.near_zero_regrets as f64 / stats.total_edges as f64
        } else {
            0.0
        };

        let determinism_ratio = if stats.infosets > 0 {
            stats.high_confidence_actions as f64 / stats.infosets as f64
        } else {
            0.0
        };

        let avg_entropy = if stats.infosets > 0 {
            stats.total_entropy / stats.infosets as f64
        } else {
            0.0
        };

        log::info!("------------------ BLUEPRINT STATS ------------------");
        log::info!("NOTE: Statistics based on first {} of {} infosets", processed, total_infosets);
        log::info!("InfoSets processed: {}", stats.infosets);
        log::info!(
            "Edges / set:   min {}  max {}  avg {:.2}",
            stats.min_edges,
            stats.max_edges,
            avg_edges
        );
        log::info!("Policy range:  [{:.4}, {:.4}]", stats.policy_min, stats.policy_max);
        log::info!("Regret range:  [{:.4}, {:.4}]", stats.regret_min, stats.regret_max);
        log::info!(
            "Zero regrets:  {} ({:.2}% of edges)",
            stats.zero_regret_count,
            100.0 * stats.zero_regret_count as f64 / stats.total_edges as f64
        );
        log::info!("Regret distribution:");
        log::info!("  < -1M:       {} edges", stats.regret_buckets[0]);
        log::info!("  [-1M, -10k): {} edges", stats.regret_buckets[1]);
        log::info!("  [-10k, -100): {} edges", stats.regret_buckets[2]);
        log::info!("  [-100, 0):   {} edges", stats.regret_buckets[3]);
        log::info!("  [0, 100):    {} edges", stats.regret_buckets[4]);
        log::info!("  [100, 10k):  {} edges", stats.regret_buckets[5]);
        log::info!("  [10k, 1M):   {} edges", stats.regret_buckets[6]);
        log::info!("  >= 1M:       {} edges", stats.regret_buckets[7]);
        log::info!("Policy weight distribution:");
        log::info!("  [0, 1):      {} edges", stats.policy_buckets[0]);
        log::info!("  [1, 10):     {} edges", stats.policy_buckets[1]);
        log::info!("  [10, 100):   {} edges", stats.policy_buckets[2]);
        log::info!("  [100, 1k):   {} edges", stats.policy_buckets[3]);
        log::info!("  >= 1k:       {} edges", stats.policy_buckets[4]);

        // Add edge count distribution
        log::info!("Edge count distribution:");
        let mut cumulative = 0usize;
        for (count, freq) in stats.edge_count_dist.iter().enumerate() {
            if *freq > 0 {
                cumulative += freq;
                let pct = *freq as f64 / stats.infosets as f64 * 100.0;
                let cum_pct = cumulative as f64 / stats.infosets as f64 * 100.0;
                log::info!(
                    "  {} edges: {:>10} infosets ({:>5.1}%, cum {:>5.1}%)",
                    count,
                    freq,
                    pct,
                    cum_pct
                );
            }
        }

        // Convergence metrics
        log::info!("Convergence metrics:");
        log::info!("  Near-zero regrets: {:.1}%", convergence_ratio * 100.0);
        log::info!(
            "  High-confidence actions: {:.1}%",
            determinism_ratio * 100.0
        );
        log::info!("  Average entropy: {:.3}", avg_entropy);

        // Additional derived metrics
        let negative_regrets: usize = stats.regret_buckets[0..4].iter().sum();
        let positive_regrets: usize = stats.regret_buckets[4..].iter().sum();
        let total_nonzero = negative_regrets + positive_regrets;
        if total_nonzero > 0 {
            let balance_ratio = negative_regrets as f64 / total_nonzero as f64;
            log::info!(
                "  Negative/Total ratio: {:.1}% (ideal ~50%)",
                balance_ratio * 100.0
            );
        }

        log::info!("-----------------------------------------------------");
    }

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
                .or_insert_with(|| RwLock::new(CompactBucket::new()));
            let mut bucket = bucket_mutex.value().write();
            for _ in 0..edges_per_bucket {
                let edge = Edge::random();
                let policy: f32 = rand::thread_rng().gen_range(0.0..1.0);
                let regret: f32 = rand::thread_rng().gen_range(-1.0..1.0);
                bucket.push((u8::from(edge), (f16::from_f32(policy), regret)));
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

        let path_str = Profile::path(Street::random());
        let path = std::path::Path::new(&path_str);

        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("Failed to create parent directories");
        }

        // Calculate total records first
        let total_records: usize = self
            .encounters
            .iter()
            .map(|entry| entry.value().read().len())
            .sum();

        log::info!(
            "Saving blueprint to {} ({} records)",
            path_str,
            total_records
        );

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

        // Define encodings - use plain encoding to minimize memory usage
        let encodings = vec![
            vec![Encoding::Plain], // history
            vec![Encoding::Plain], // present
            vec![Encoding::Plain], // futures
            vec![Encoding::Plain], // edge
            vec![Encoding::Plain], // regret
            vec![Encoding::Plain], // policy
        ];

        // Write options - optimized for speed while keeping compression
        let options = WriteOptions {
            write_statistics: true, // Keep statistics for better query performance
            compression: CompressionOptions::Zstd(Some(
                arrow2::io::parquet::write::ZstdLevel::try_new(1).unwrap(),
            )), // Fast ZSTD compression level 1
            version: Version::V2,
            data_pagesize_limit: Some(2 * 1024 * 1024), // 2MB pages for good balance
        };

        // Create output file and writer with large buffer for better I/O performance
        let file = File::create(path).expect(&format!("Failed to create file at {}", path_str));
        let writer_inner = BufWriter::with_capacity(32 * 1024 * 1024, file); // 32MB buffer
        let mut writer = FileWriter::try_new(writer_inner, schema.clone(), options)
            .expect("Failed to create parquet writer");

        const RECORDS_PER_CHUNK: usize = 5_000_000; // 3M records per chunk
        let written_records = std::sync::atomic::AtomicUsize::new(0);

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
                let edge_enum: Edge = edge.into();
                chunk_histories.push(u64::from(*bucket.history()) as i64);
                chunk_presents.push(u64::from(*bucket.present()) as i64);
                chunk_futures.push(u64::from(*bucket.futures()) as i64);
                chunk_edges.push(u64::from(edge_enum) as u32);
                chunk_regrets.push(regret);
                chunk_policies.push(f32::from(policy));

                let current_count = written_records.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

                // Update progress less frequently to reduce overhead
                if current_count % 1_000_000 == 0 {
                    progress.inc(1_000_000);
                }

                // Write chunk when it's full
                if chunk_histories.len() >= RECORDS_PER_CHUNK {
                    Profile::write_parquet_chunk(
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

                    // Reuse vectors instead of reallocating - just reset length
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
            Profile::write_parquet_chunk(
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
        )
        .expect("Failed to create row group iterator");

        // Write this chunk as a row group
        for group in row_groups {
            writer
                .write(group.expect("Failed to get row group"))
                .expect("Failed to write row group");
        }
    }

    /// Load profile data from Apache Parquet format.
    #[cfg(feature = "native")]
    pub fn load_parquet() -> Profile {
        use crate::save::disk::Disk;
        use rayon::prelude::*;
        use std::fs::File;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let path_str = Profile::path(Street::random());
        log::info!("{:<32}{:<32}", "loading     blueprint", path_str);

        let mut file =
            File::open(&path_str).expect(&format!("Failed to open blueprint file: {}", path_str));

        // Read metadata
        let metadata = read_metadata(&mut file).expect("Failed to read parquet metadata");

        // Infer schema
        let schema = infer_schema(&metadata).expect("Failed to infer schema");

        let total_rows: usize = metadata
            .row_groups
            .iter()
            .map(|rg| rg.num_rows() as usize)
            .sum();
        log::info!(
            "Loading blueprint from {} row groups ({} total rows)",
            metadata.row_groups.len(),
            total_rows
        );

        let progress = crate::progress(total_rows);
        progress.set_message("Loading blueprint from parquet");

        let encounters: DashMap<Info, RwLock<CompactBucket>, FxBuildHasher> =
            DashMap::with_hasher(FxBuildHasher::default());

        // Atomic row counter for optional logging/debugging
        let processed_rows = AtomicUsize::new(0);

        // Share path so each thread can reopen the file independently
        let path_shared = Arc::new(path_str.clone());

        // Share schema across threads to avoid cloning
        let schema_shared = Arc::new(schema.clone());

        // Build a dedicated thread pool for loading
        rayon::ThreadPoolBuilder::new()
            .num_threads(6)
            .build()
            .expect("Failed to build rayon pool")
            .install(|| {
                metadata
                    .row_groups
                    .par_iter()
                    .enumerate()
                    .for_each(|(_rg_idx, row_group)| {
                        // Open a fresh handle for this row-group
                        let file_handle = std::fs::File::open(&*path_shared)
                            .expect("Failed to reopen parquet file");

                        // Use buffered reader for better I/O performance
                        let buffered_reader = std::io::BufReader::with_capacity(
                            8 * 1024 * 1024, // 8MB buffer per thread
                            file_handle
                        );

                                                let chunks = FileReader::new(
                            buffered_reader,
                            vec![row_group.clone()],
                            schema_shared.as_ref().clone(),
                            None, // projection - read all columns
                            None, // limit
                            None, // page filter - not using chunk size limit here due to API constraints
                        );

                        for chunk_result in chunks {
                            let chunk = chunk_result.expect("Failed to read chunk");

                            // Column arrays
                            let arrays = chunk.arrays();
                            debug_assert_eq!(arrays.len(), 6, "Unexpected column count");

                            let histories = arrays[0]
                                .as_any()
                                .downcast_ref::<Int64Array>()
                                .expect("history col");
                            let presents = arrays[1]
                                .as_any()
                                .downcast_ref::<Int64Array>()
                                .expect("present col");
                            let futures = arrays[2]
                                .as_any()
                                .downcast_ref::<Int64Array>()
                                .expect("futures col");
                            let edges = arrays[3]
                                .as_any()
                                .downcast_ref::<UInt32Array>()
                                .expect("edge col");
                            let regrets = arrays[4]
                                .as_any()
                                .downcast_ref::<Float32Array>()
                                .expect("regret col");
                            let policies = arrays[5]
                                .as_any()
                                .downcast_ref::<Float32Array>()
                                .expect("policy col");

                            let num_rows = chunk.len();

                            // Process rows more efficiently in batches
                            const BATCH_SIZE: usize = 1000;
                            for batch_start in (0..num_rows).step_by(BATCH_SIZE) {
                                let batch_end = (batch_start + BATCH_SIZE).min(num_rows);

                                for i in batch_start..batch_end {
                                    let history = Path::from(histories.value(i) as u64);
                                    let present = Abstraction::from(presents.value(i) as u64);
                                    let futures = Path::from(futures.value(i) as u64);
                                    let edge = Edge::from(edges.value(i) as u64);
                                    let regret = regrets.value(i);
                                    let policy = policies.value(i);

                                    let info = Info::from((history, present, futures));
                                    let bucket_mutex = encounters
                                        .entry(info)
                                        .or_insert_with(|| RwLock::new(CompactBucket::new()));
                                    bucket_mutex
                                        .value()
                                        .write()
                                        .push((u8::from(edge), (f16::from_f32(policy), regret)));
                                }

                                // Update progress in batches to reduce overhead
                                let batch_size = batch_end - batch_start;
                                processed_rows.fetch_add(batch_size, Ordering::Relaxed);
                                progress.inc(batch_size as u64);
                            }
                        }
                    });
            });

        progress.finish_with_message("Blueprint loaded from parquet");

        let profile = Profile {
            encounters,
            iterations: 0,
            regret_min: crate::REGRET_MIN,
            regret_max: crate::REGRET_MAX,
        };

        // Always log aggregate statistics after loading
        #[cfg(feature = "native")]
        profile.log_stats();

        profile
    }

    /// Load profile data from legacy zstd-compressed PostgreSQL binary format files.
    /// This method handles the original file format used before the mmap optimization.
    #[cfg(feature = "native")]
    fn load_legacy_pgcopy() -> Profile {
        use crate::save::disk::Disk;
        use byteorder::{ByteOrder, BE};
        use std::fs::File;
        use std::io::{BufReader, Read};

        let path = Profile::path(Street::random());
        log::info!("{:<32}{:<32}", "loading     blueprint (legacy)", path);

        // Open file and create zstd decoder (we know it's zstd compressed)
        let file = File::open(&path).expect("Failed to open blueprint file");
        let reader_inner = ZDecoder::new(file).expect("Failed to create zstd decoder");
        let mut reader = BufReader::with_capacity(1024 * 1024, reader_inner);

        // Skip the PostgreSQL binary COPY header
        let mut skip = [0u8; PGCOPY_HEADER_SIZE];
        reader
            .read_exact(&mut skip)
            .expect("Failed to skip pgcopy header");

        let encounters: DashMap<Info, RwLock<CompactBucket>, FxBuildHasher> =
            DashMap::with_hasher(FxBuildHasher::default());
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
            reader
                .read_exact(&mut record)
                .expect("Failed to read record bytes");

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
            let regret = if regret_raw.is_finite() {
                regret_raw
            } else {
                0.0
            };
            let policy = if policy_raw.is_finite() {
                policy_raw
            } else {
                0.0
            };

            let info = Info::from((history, present, futures));
            let bucket_mutex = encounters
                .entry(info)
                .or_insert_with(|| RwLock::new(CompactBucket::new()));
            let mut bucket = bucket_mutex.value().write();
            bucket.push((u8::from(edge), (f16::from_f32(policy), regret)));
        }

        let profile = Profile {
            encounters,
            iterations: 0,
            regret_min: crate::REGRET_MIN,
            regret_max: crate::REGRET_MAX,
        };

        // Log stats after loading legacy format as well
        #[cfg(feature = "native")]
        profile.log_stats();

        profile
    }
}

#[cfg(feature = "native")]
impl crate::save::upload::Table for Profile {
    fn name() -> String {
        Profile::name()
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
        vec![Profile::path(Street::random())]
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
        Profile::name()
    }
    fn grow(_: Street) -> Self {
        unreachable!("must be learned in MCCFR minimization")
    }
    fn path(_: Street) -> String {
        let current_dir = std::env::current_dir().unwrap_or_default();
        let path = PathBuf::from(current_dir).join("pgcopy").join(Profile::name());

        path.to_string_lossy().into_owned()
    }
    fn done(_: Street) -> bool {
        std::fs::metadata(Profile::path(Street::random())).is_ok()
    }
    fn load(_: Street) -> Self {
        use std::fs::File;
        use std::io::Read;

        let path = Profile::path(Street::random());
        log::info!("{:<32}{:<32}", "loading     blueprint", path);

        // Detect format by checking for zstd magic bytes
        let mut magic = [0u8; 4];
        let mut file = File::open(&path).expect("Failed to open blueprint file");
        file.read_exact(&mut magic)
            .expect("Failed to read file header");

        if magic == ZSTD_MAGIC {
            log::debug!("Detected zstd-compressed PostgreSQL binary format (legacy)");
            Profile::load_legacy_pgcopy()
        } else {
            log::debug!("Detected parquet format");
            Profile::load_parquet()
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