

#[cfg(feature = "native")]
use std::fs::File;
#[cfg(feature = "native")]
use std::io::{Read, Write, Seek};
#[cfg(feature = "native")]
use memmap2::MmapOptions;
#[cfg(feature = "native")]
use byteorder::{ByteOrder, LittleEndian, BigEndian, WriteBytesExt};
#[cfg(feature = "native")]
use zstd::stream::Encoder as ZEncoder;
#[cfg(feature = "native")]
use arrow2::{
    array::{Int64Array, UInt32Array, Float32Array, Array},
    chunk::Chunk,
    datatypes::{DataType, Field, Schema},
    io::parquet::write::{
        CompressionOptions, Encoding, FileWriter, RowGroupIterator,
        Version, WriteOptions
    },
};
#[cfg(feature = "native")]
use std::sync::Arc;

// File format constants (from profile.rs)
const ZSTD_MAGIC: [u8; 4] = [0x28, 0xB5, 0x2F, 0xFD];
const MMAP_RECORD_SIZE: usize = 40; // 8+8+8+8+4+4 bytes per record
const MMAP_HEADER_SIZE: usize = 8; // record count header
const CHUNK_SIZE: usize = 1000; // Process records in chunks

/// Converter for converting between blueprint file formats
pub struct Converter;

impl Converter {
    /// Convert a blueprint file to PostgreSQL-compatible format.
    ///
    /// This method reads directly from the blueprint file using memory mapping
    /// and processes it in chunks to avoid loading the entire file into memory.
    /// Works with both legacy (zstd PostgreSQL) and new (mmap) formats.
    ///
    /// # Output Format
    /// - File: `blueprint.pg` in the current directory
    /// - Format: zstd-compressed PostgreSQL binary COPY format
    /// - Columns: past, present, future, edge, regret, policy
    ///
    /// # Memory Efficiency
    /// This method processes the file in chunks without loading the entire
    /// profile into memory, making it suitable for large files on memory-constrained systems.
    ///
    /// # Example
    /// ```bash
    /// cargo run --release -- --convert-blueprint-to-postgres
    /// ```
    #[cfg(feature = "native")]
    pub fn convert_blueprint_to_postgres() {
        log::info!("Starting blueprint to PostgreSQL format conversion");

        // Look for the old blueprint file without extension
        let current_dir = std::env::current_dir().unwrap_or_default();
        let old_blueprint_path = current_dir.join("pgcopy").join("blueprint");

        if !old_blueprint_path.exists() {
            log::error!("Blueprint file not found at: {}", old_blueprint_path.display());
            log::error!("Please ensure a blueprint file exists before running conversion");
            return;
        }

        log::info!("Reading blueprint file from: {}", old_blueprint_path.display());

        // Detect the format by checking magic bytes
        Self::convert_blueprint(&old_blueprint_path.to_string_lossy());

        log::info!("Blueprint conversion completed");
    }

    /// Convert blueprint file, handling both legacy and new formats
    #[cfg(feature = "native")]
    fn convert_blueprint(source_path: &str) {
        let format = Self::detect_format(source_path);

        match format {
            FormatType::LegacyZstdPostgres => {
                log::info!("Source file is already in PostgreSQL format - no conversion needed");
            }
            FormatType::NewMmap => {
                log::info!("Converting from new mmap format to PostgreSQL format");
                Self::convert_mmap_to_postgres(source_path);
            }
            FormatType::Parquet => {
                log::info!("Source file is already in Parquet format - no conversion needed");
            }
        }
    }

    /// Convert blueprint file to Parquet format
    #[cfg(feature = "native")]
    pub fn convert_to_parquet(source_path: &str) {
        let format = Self::detect_format(source_path);

        match format {
            FormatType::LegacyZstdPostgres => {
                log::error!("Converting from PostgreSQL format to Parquet is not yet implemented");
                // TODO: Implement if needed
            }
            FormatType::NewMmap => {
                log::info!("Converting from mmap format to Parquet format");
                Self::convert_mmap_to_parquet(source_path);
            }
            FormatType::Parquet => {
                log::info!("Source file is already in Parquet format - no conversion needed");
            }
        }
    }

    /// Detect the format of the blueprint file
    #[cfg(feature = "native")]
    fn detect_format(path: &str) -> FormatType {
        let mut file = File::open(path)
            .expect("Failed to open source file");
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)
            .expect("Failed to read file header");

        if magic == ZSTD_MAGIC {
            FormatType::LegacyZstdPostgres
        } else if magic == [b'P', b'A', b'R', b'1'] {
            // Parquet magic bytes at the beginning
            FormatType::Parquet
        } else {
            // Check for Parquet magic bytes at the end (more common)
            use std::io::SeekFrom;
            if let Ok(metadata) = file.metadata() {
                let file_size = metadata.len();
                if file_size >= 4 {
                    let mut end_magic = [0u8; 4];
                    if file.seek(SeekFrom::End(-4)).is_ok() &&
                       file.read_exact(&mut end_magic).is_ok() &&
                       end_magic == [b'P', b'A', b'R', b'1'] {
                        return FormatType::Parquet;
                    }
                }
            }
            FormatType::NewMmap
        }
    }

                /// Convert new mmap format to PostgreSQL format in chunks
    #[cfg(feature = "native")]
    fn convert_mmap_to_postgres(source_path: &str) {
        log::info!("Converting mmap format to PostgreSQL format");

        // Memory-map the source file
        let source_file = File::open(source_path)
            .expect("Failed to open source file");

        let mmap = unsafe { MmapOptions::new().map(&source_file) }
            .expect("Failed to memory map source file");

        // Validate file size and read header
        if mmap.len() < MMAP_HEADER_SIZE {
            panic!("Source file too small to contain header: {} bytes", mmap.len());
        }

        let total_records = LittleEndian::read_u64(&mmap[0..MMAP_HEADER_SIZE]) as usize;
        let expected_size = MMAP_HEADER_SIZE + (total_records * MMAP_RECORD_SIZE);

        if mmap.len() < expected_size {
            panic!("Source file size mismatch: expected {} bytes, got {}", expected_size, mmap.len());
        }

        log::info!("Converting {} records from mmap to PostgreSQL format", total_records);
        
        let progress = crate::progress(total_records);
        progress.set_message("Converting mmap to PostgreSQL");

        // Determine output path - use pgcopy directory for PostgreSQL format
        let current_dir = std::env::current_dir().unwrap_or_default();
        let pgcopy_dir = current_dir.join("pgcopy");
        std::fs::create_dir_all(&pgcopy_dir).expect("Failed to create pgcopy directory");
        let output_path = pgcopy_dir.join("blueprint.pg");

        // Create output file with zstd compression
        let output_file = File::create(&output_path)
            .expect("Failed to create output file blueprint.pg");

        let mut encoder = ZEncoder::new(output_file, 3)
            .expect("Failed to create zstd encoder");

        // Write PostgreSQL binary COPY header
        Self::write_pg_header(&mut encoder);

                                // Process records in chunks
        for chunk_start in (0..total_records).step_by(CHUNK_SIZE) {
            let chunk_end = (chunk_start + CHUNK_SIZE).min(total_records);

            for i in chunk_start..chunk_end {
                let offset = MMAP_HEADER_SIZE + (i * MMAP_RECORD_SIZE);

                // Read record from mmap (little-endian format)
                let history = LittleEndian::read_u64(&mmap[offset..offset+8]);
                let present = LittleEndian::read_u64(&mmap[offset+8..offset+16]);
                let futures = LittleEndian::read_u64(&mmap[offset+16..offset+24]);
                let edge = LittleEndian::read_u64(&mmap[offset+24..offset+32]);
                let regret = LittleEndian::read_f32(&mmap[offset+32..offset+36]);
                let policy = LittleEndian::read_f32(&mmap[offset+36..offset+40]);

                // Write PostgreSQL record (big-endian format)
                Self::write_pg_record(&mut encoder, history, present, futures, edge, regret, policy);

                progress.inc(1);
            }
        }

        // Write PostgreSQL binary trailer
        Self::write_pg_trailer(&mut encoder);

        // Finalize compression
        encoder.finish()
            .expect("Failed to finalize zstd compression");

        progress.finish_with_message("Conversion complete");
    }

        /// Write PostgreSQL binary COPY header
    #[cfg(feature = "native")]
    fn write_pg_header<W: Write>(writer: &mut W) {
        // PostgreSQL binary format signature
        writer.write_all(b"PGCOPY\n\xff\r\n\0").expect("Failed to write PostgreSQL signature");
        // Flags field (32-bit integer, no OIDs)
        writer.write_u32::<BigEndian>(0).expect("Failed to write flags field");
        // Header extension length (32-bit integer, no extensions)
        writer.write_u32::<BigEndian>(0).expect("Failed to write header extension length");
    }

    /// Write a single PostgreSQL binary record
    #[cfg(feature = "native")]
    fn write_pg_record<W: Write>(
        writer: &mut W,
        history: u64,
        present: u64,
        futures: u64,
        edge: u64,
        regret: f32,
        policy: f32,
    ) {
        // Number of fields (16-bit integer)
        writer.write_u16::<BigEndian>(6).expect("Failed to write field count");

        // Field 1: past (history) - 8 bytes
        writer.write_u32::<BigEndian>(8).expect("Failed to write field length"); // field length
        writer.write_u64::<BigEndian>(history).expect("Failed to write history");

        // Field 2: present - 8 bytes
        writer.write_u32::<BigEndian>(8).expect("Failed to write field length"); // field length
        writer.write_u64::<BigEndian>(present).expect("Failed to write present");

        // Field 3: future (futures) - 8 bytes
        writer.write_u32::<BigEndian>(8).expect("Failed to write field length"); // field length
        writer.write_u64::<BigEndian>(futures).expect("Failed to write futures");

        // Field 4: edge - 8 bytes
        writer.write_u32::<BigEndian>(8).expect("Failed to write field length"); // field length
        writer.write_u64::<BigEndian>(edge).expect("Failed to write edge");

        // Field 5: regret - 4 bytes
        writer.write_u32::<BigEndian>(4).expect("Failed to write field length"); // field length
        writer.write_f32::<BigEndian>(regret).expect("Failed to write regret");

        // Field 6: policy - 4 bytes
        writer.write_u32::<BigEndian>(4).expect("Failed to write field length"); // field length
        writer.write_f32::<BigEndian>(policy).expect("Failed to write policy");
    }

    /// Write PostgreSQL binary COPY trailer
    #[cfg(feature = "native")]
    fn write_pg_trailer<W: Write>(writer: &mut W) {
        // File trailer (16-bit integer, -1 indicates end of data)
        writer.write_i16::<BigEndian>(-1).expect("Failed to write PostgreSQL trailer");
    }

    /// Convert mmap format to Parquet format using streaming chunks
    #[cfg(feature = "native")]
    fn convert_mmap_to_parquet(source_path: &str) {
        use std::io::BufWriter;

        log::info!("Converting mmap format to Parquet format");

        // Memory-map the source file
        let source_file = File::open(source_path)
            .expect("Failed to open source file");

        let mmap = unsafe { MmapOptions::new().map(&source_file) }
            .expect("Failed to memory map source file");

        // Validate file size and read header
        if mmap.len() < MMAP_HEADER_SIZE {
            panic!("Source file too small to contain header: {} bytes", mmap.len());
        }

        let total_records = LittleEndian::read_u64(&mmap[0..MMAP_HEADER_SIZE]) as usize;
        let expected_size = MMAP_HEADER_SIZE + (total_records * MMAP_RECORD_SIZE);

        if mmap.len() < expected_size {
            panic!("Source file size mismatch: expected {} bytes, got {}", expected_size, mmap.len());
        }

        log::info!("Converting {} records from mmap to Parquet format", total_records);
        
        let progress = crate::progress(total_records);
        progress.set_message("Converting mmap to parquet");

        // Determine output path - use pgcopy directory
        let current_dir = std::env::current_dir().unwrap_or_default();
        let pgcopy_dir = current_dir.join("pgcopy");
        std::fs::create_dir_all(&pgcopy_dir).expect("Failed to create pgcopy directory");
        let output_path = pgcopy_dir.join("blueprint.parquet");

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
            compression: CompressionOptions::Lz4, // Much faster than zstd
            version: Version::V2,
            data_pagesize_limit: Some(1024 * 1024), // Larger 1MB pages for better throughput
        };

        // Create output file and writer with larger buffer for better performance
        let file = File::create(&output_path)
            .expect(&format!("Failed to create file at {}", output_path.display()));
        let writer_inner = BufWriter::with_capacity(8 * 1024 * 1024, file); // 8MB buffer
        let mut writer = FileWriter::try_new(writer_inner, schema.clone(), options)
            .expect("Failed to create parquet writer");

        // Process records in chunks to avoid loading everything into memory
        const RECORDS_PER_CHUNK: usize = 1_000_000; // Process 1M records at a time for better performance

        for chunk_start in (0..total_records).step_by(RECORDS_PER_CHUNK) {
            let chunk_end = (chunk_start + RECORDS_PER_CHUNK).min(total_records);
            let chunk_size = chunk_end - chunk_start;

            // Allocate vectors for this chunk only
            let mut histories = Vec::with_capacity(chunk_size);
            let mut presents = Vec::with_capacity(chunk_size);
            let mut futures_vec = Vec::with_capacity(chunk_size);
            let mut edges = Vec::with_capacity(chunk_size);
            let mut regrets = Vec::with_capacity(chunk_size);
            let mut policies = Vec::with_capacity(chunk_size);

            // Read chunk of records from mmap
            for i in chunk_start..chunk_end {
                let offset = MMAP_HEADER_SIZE + (i * MMAP_RECORD_SIZE);

                // Read record from mmap (little-endian format)
                let history = LittleEndian::read_u64(&mmap[offset..offset+8]);
                let present = LittleEndian::read_u64(&mmap[offset+8..offset+16]);
                let futures = LittleEndian::read_u64(&mmap[offset+16..offset+24]);
                let edge = LittleEndian::read_u64(&mmap[offset+24..offset+32]);
                let regret = LittleEndian::read_f32(&mmap[offset+32..offset+36]);
                let policy = LittleEndian::read_f32(&mmap[offset+36..offset+40]);

                // Convert to appropriate types for Parquet
                histories.push(history as i64);
                presents.push(present as i64);
                futures_vec.push(futures as i64);
                edges.push(edge as u32);
                regrets.push(regret);
                policies.push(policy);

                progress.inc(1);
            }

            // Create arrow arrays for this chunk
            let hist_array = Int64Array::from_slice(&histories);
            let pres_array = Int64Array::from_slice(&presents);
            let fut_array = Int64Array::from_slice(&futures_vec);
            let edge_array = UInt32Array::from_slice(&edges);
            let regret_array = Float32Array::from_slice(&regrets);
            let policy_array = Float32Array::from_slice(&policies);

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
                &schema,
                options,
                encodings.clone(),
            ).expect("Failed to create row group iterator");

            // Write this chunk as a row group
            for group in row_groups {
                writer.write(group.expect("Failed to get row group"))
                    .expect("Failed to write row group");
            }

        }

        // Finalize the parquet file
        let _size = writer.end(None).expect("Failed to finalize parquet file");
        progress.finish_with_message("Conversion complete");
    }

    /// Convert blueprint file to Parquet format
    ///
    /// This method converts old mmap format files to the new Parquet format.
    /// If the file is already in Parquet format, no conversion is performed.
    ///
    /// # Example
    /// ```bash
    /// cargo run --release -- --convert-to-parquet
    /// ```
    #[cfg(feature = "native")]
    pub fn convert_blueprint_to_parquet() {
        log::info!("Starting blueprint to Parquet format conversion");

        // Look for the old blueprint file without extension
        let current_dir = std::env::current_dir().unwrap_or_default();
        let old_blueprint_path = current_dir.join("pgcopy").join("blueprint");

        if !old_blueprint_path.exists() {
            log::error!("Blueprint file not found at: {}", old_blueprint_path.display());
            log::error!("Please ensure a blueprint file exists before running conversion");
            return;
        }

        log::info!("Reading blueprint file from: {}", old_blueprint_path.display());

        // Convert the file
        Self::convert_to_parquet(&old_blueprint_path.to_string_lossy());

        log::info!("Blueprint conversion completed");
    }

    /// Stub implementation for non-native feature
    #[cfg(not(feature = "native"))]
    pub fn convert_blueprint_to_postgres() {
        log::error!("PostgreSQL conversion requires the 'native' feature");
    }

    /// Stub implementation for non-native feature
    #[cfg(not(feature = "native"))]
    pub fn convert_blueprint_to_parquet() {
        log::error!("Parquet conversion requires the 'native' feature");
    }
}

#[cfg(feature = "native")]
#[derive(Debug)]
enum FormatType {
    LegacyZstdPostgres,
    NewMmap,
    Parquet,
}