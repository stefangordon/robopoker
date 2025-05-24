use crate::mccfr::nlhe::profile::Profile;
use crate::cards::street::Street;
use crate::save::disk::Disk;
use crate::Arbitrary;

#[cfg(feature = "native")]
use std::fs::File;
#[cfg(feature = "native")]
use std::io::{Read, Write};
#[cfg(feature = "native")]
use std::path::Path;
#[cfg(feature = "native")]
use memmap2::MmapOptions;
#[cfg(feature = "native")]
use byteorder::{ByteOrder, LittleEndian, BigEndian, WriteBytesExt};
#[cfg(feature = "native")]
use zstd::stream::Encoder as ZEncoder;

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
    /// cargo run --release -- --generatepgdata
    /// ```
    #[cfg(feature = "native")]
    pub fn generate_pg_data() {
        log::info!("Starting blueprint to PostgreSQL format conversion");

        // Check if the blueprint file exists
        let blueprint_path = Profile::path(Street::random());
        if !std::path::Path::new(&blueprint_path).exists() {
            log::error!("Blueprint file not found at: {}", blueprint_path);
            log::error!("Please ensure a blueprint file exists before running conversion");
            return;
        }

        log::info!("Reading blueprint file from: {}", blueprint_path);

        // Detect the format by checking magic bytes
        Self::convert_blueprint(&blueprint_path);

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
        } else {
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

        // Determine output path (same directory as source)
        let source_dir = Path::new(source_path)
            .parent()
            .expect("Failed to determine source directory");
        let output_path = source_dir.join("blueprint.pg");

        // Create output file with zstd compression
        let output_file = File::create(&output_path)
            .expect("Failed to create output file blueprint.pg");

        let mut encoder = ZEncoder::new(output_file, 3)
            .expect("Failed to create zstd encoder");

        // Write PostgreSQL binary COPY header
        Self::write_pg_header(&mut encoder);

                                // Process records in chunks
        let mut records_processed = 0;

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

                records_processed += 1;
            }

            // Log progress every chunk
            let percentage = (records_processed * 100) / total_records;
            log::info!("Conversion progress: {}% ({} / {} records)", percentage, records_processed, total_records);
        }

        // Write PostgreSQL binary trailer
        Self::write_pg_trailer(&mut encoder);

        // Finalize compression
        encoder.finish()
            .expect("Failed to finalize zstd compression");

        log::info!("Successfully converted {} records to {}", records_processed, output_path.display());
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

    /// Stub implementation for non-native feature
    #[cfg(not(feature = "native"))]
    pub fn generate_pg_data() {
        log::error!("PostgreSQL data generation requires the 'native' feature");
    }
}

#[cfg(feature = "native")]
#[derive(Debug)]
enum FormatType {
    LegacyZstdPostgres,
    NewMmap,
}