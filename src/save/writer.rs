use super::derive::Derive;
use super::upload::Table;
use crate::cards::street::Street;
use crate::clustering::abstraction::Abstraction;
use crate::clustering::metric::Metric;
use crate::clustering::transitions::Decomp;
use crate::mccfr::nlhe::encoder::Encoder;
use crate::mccfr::nlhe::profile::Profile;
use byteorder::ReadBytesExt;
use byteorder::BE;
use futures::pin_mut;
use std::fs::File;
use std::io::BufReader;
use std::io::Read;
use std::io::Seek;
use std::io::SeekFrom;
use std::sync::Arc;
use tokio_postgres::binary_copy::BinaryCopyInWriter;
use tokio_postgres::types::ToSql;
use tokio_postgres::Client;
use tokio_postgres::Error as E;
use zstd::stream::Decoder as ZDecoder;

pub struct Writer(Arc<Client>);

impl From<Arc<Client>> for Writer {
    fn from(client: Arc<Client>) -> Self {
        Self(client)
    }
}

impl Writer {
    pub async fn publish() -> Result<(), E> {
        let postgres = Self(crate::db().await);

        // Optimize PostgreSQL settings for bulk loading
        // - Disable synchronous commits
        // - Use unlogged tables during load (converted to logged after)
        postgres
            .0
            .batch_execute("SET synchronous_commit = OFF;")
            .await?;

        // Upload in dependency order
        postgres.upload::<Metric>().await?;
        postgres.upload::<Decomp>().await?;
        postgres.upload::<Encoder>().await?;
        postgres.upload::<Profile>().await?; // This is the slow one

        // Derive tables
        postgres.derive::<Abstraction>().await?;
        postgres.derive::<Street>().await?;

        // Analyze tables
        postgres.0.batch_execute("VACUUM ANALYZE;").await?;
        Ok(())
    }

    async fn upload<T>(&self) -> Result<(), E>
    where
        T: Table,
    {
        let ref name = T::name();
        if self.absent(name).await? {
            log::info!("creating table ({})", name);
            self.0.batch_execute(&T::creates()).await?;
            self.0.batch_execute(&T::truncates()).await?;
        }
        if self.vacant(name).await? {
            log::info!("copying {}", name);

            // Use optimized stream for blueprint table
            if name == "blueprint" {
                self.stream_optimized::<T>().await?;
            } else {
                self.stream::<T>().await?;
            }

            // Create indices CONCURRENTLY for large tables to avoid locking
            if name == "blueprint" || name == "isomorphism" {
                log::info!("creating indices for {} (this may take a while)", name);
                // Parse and execute each index creation separately for CONCURRENTLY support
                for index_stmt in T::indices().split(';').filter(|s| !s.trim().is_empty()) {
                    let concurrent_stmt = index_stmt.trim().replace(
                        "CREATE INDEX IF NOT EXISTS",
                        "CREATE INDEX CONCURRENTLY IF NOT EXISTS",
                    );
                    if concurrent_stmt.contains("CREATE INDEX") {
                        // Execute index creation with lower priority
                        self.0.batch_execute("SET vacuum_cost_delay = 10;").await?;
                        self.0.batch_execute(&concurrent_stmt).await?;
                    } else {
                        // Non-index statements (like INSERTs for metric table)
                        self.0.batch_execute(index_stmt.trim()).await?;
                    }
                }
            } else {
                self.0.batch_execute(&T::indices()).await?;
            }
            Ok(())
        } else {
            log::info!("table data already uploaded ({})", name);
            Ok(())
        }
    }

    async fn derive<D>(&self) -> Result<(), E>
    where
        D: Derive,
    {
        let ref name = D::name();
        if self.absent(name).await? {
            log::info!("creating table ({})", name);
            self.0.batch_execute(&D::creates()).await?;
        }
        if self.vacant(name).await? {
            log::info!("deriving {}", name);
            self.0.batch_execute(&D::indexes()).await?;
            self.0.batch_execute(&D::derives()).await?;
            Ok(())
        } else {
            log::info!("table data already derived  ({})", name);
            Ok(())
        }
    }

    /// Optimized streaming for large tables (blueprint)
    async fn stream_optimized<T>(&self) -> Result<(), E>
    where
        T: Table,
    {
        let sink = self.0.copy_in(&T::copy()).await?;
        let writer = BinaryCopyInWriter::new(sink, T::columns());
        pin_mut!(writer);

        // Much larger batch size for blueprint table
        const LARGE_BATCH: usize = 50_000; // Increased from 8,192
        let cols = T::columns().len();
        let mut buffer: Vec<Vec<Field>> = Vec::with_capacity(LARGE_BATCH);

        let mut total_rows: usize = 0;
        let start_time = std::time::Instant::now();
        let mut last_log_time = start_time;

        let mut tag = [0u8; 2];

        for src in T::sources() {
            log::info!("Processing source file: {}", src);

            // Detect compression
            let mut magic = [0u8; 4];
            let mut file_for_magic = File::open(&src).expect("file not found");
            let _ = file_for_magic.read_exact(&mut magic);
            file_for_magic.seek(SeekFrom::Start(0)).unwrap();

            // Use larger buffer for reading
            let reader_inner: Box<dyn Read> = if magic == [0x28, 0xB5, 0x2F, 0xFD] {
                log::debug!("Detected zstd compression");
                Box::new(ZDecoder::new(file_for_magic).expect("zstd decode"))
            } else {
                log::debug!("Using uncompressed file");
                Box::new(file_for_magic)
            };

            // Increase buffer size for better I/O performance
            let mut reader = BufReader::with_capacity(4 * 1024 * 1024, reader_inner);

            // Skip PostgreSQL header
            let mut skip = [0u8; 19];
            reader.read_exact(&mut skip).expect("skip pgcopy header");

            loop {
                if let Err(_) = reader.read_exact(&mut tag) {
                    break;
                }

                let fields = u16::from_be_bytes(tag);
                if fields == 0xFFFF {
                    break;
                }

                if fields as usize != cols {
                    // Skip invalid row
                    for _ in 0..fields {
                        let len = reader.read_u32::<BE>().unwrap_or(0);
                        reader
                            .by_ref()
                            .take(len as u64)
                            .read_to_end(&mut Vec::new())
                            .ok();
                    }
                    continue;
                }

                let mut row: Vec<Field> = Vec::with_capacity(cols);
                for ty in T::columns() {
                    let len = reader.read_u32::<BE>().unwrap();
                    let _ = len;
                    match *ty {
                        tokio_postgres::types::Type::INT8 => {
                            let val = reader.read_i64::<BE>().unwrap();
                            row.push(Field::I64(val));
                        }
                        tokio_postgres::types::Type::FLOAT4 => {
                            let val = reader.read_f32::<BE>().unwrap();
                            row.push(Field::F32(val));
                        }
                        _ => unreachable!("unsupported column type"),
                    }
                }

                buffer.push(row);
                total_rows += 1;

                if buffer.len() == LARGE_BATCH {
                    // Write batch
                    for row in &buffer {
                        writer.as_mut().write_raw(row.iter()).await?;
                    }
                    buffer.clear();

                    // Enhanced progress logging
                    let now = std::time::Instant::now();
                    if now.duration_since(last_log_time).as_secs() >= 5 {
                        let elapsed = now.duration_since(start_time).as_secs_f64();
                        let rows_per_sec = total_rows as f64 / elapsed;
                        log::info!(
                            "COPY progress: {} rows ({:.0} rows/sec, {:.1} MB/sec)",
                            total_rows,
                            rows_per_sec,
                            (rows_per_sec * 40.0) / 1_048_576.0 // 40 bytes per row
                        );
                        last_log_time = now;
                    }
                }
            }
        }

        // Write remaining rows
        for row in &buffer {
            writer.as_mut().write_raw(row.iter()).await?;
        }

        writer.finish().await?;

        let elapsed = start_time.elapsed().as_secs_f64();
        log::info!(
            "COPY completed: {} total rows in {:.1}s ({:.0} rows/sec)",
            total_rows,
            elapsed,
            total_rows as f64 / elapsed
        );

        Ok(())
    }

    /// Original stream method for smaller tables
    async fn stream<T>(&self) -> Result<(), E>
    where
        T: Table,
    {
        let sink = self.0.copy_in(&T::copy()).await?;
        let writer = BinaryCopyInWriter::new(sink, T::columns());
        pin_mut!(writer);

        // Original batch size for smaller tables
        const BATCH: usize = 8_192;
        let cols = T::columns().len();
        let mut buffer: Vec<Vec<Field>> = Vec::with_capacity(BATCH);

        let mut batch_count: usize = 0;
        let mut tag = [0u8; 2];

        for src in T::sources() {
            // Detect zstd compression
            let mut magic = [0u8; 4];
            let mut file_for_magic = File::open(&src).expect("file not found");
            let _ = file_for_magic.read_exact(&mut magic);
            file_for_magic.seek(SeekFrom::Start(0)).unwrap();

            let reader_inner: Box<dyn Read> = if magic == [0x28, 0xB5, 0x2F, 0xFD] {
                log::debug!("Detected zstd compression in file: {}", src);
                Box::new(ZDecoder::new(file_for_magic).expect("zstd decode"))
            } else {
                log::debug!("Using uncompressed file: {}", src);
                Box::new(file_for_magic)
            };

            let mut reader = BufReader::new(reader_inner);

            // Skip PostgreSQL header
            let mut skip = [0u8; 19];
            reader.read_exact(&mut skip).expect("skip pgcopy header");

            loop {
                if let Err(_) = reader.read_exact(&mut tag) {
                    break;
                }

                let fields = u16::from_be_bytes(tag);
                if fields == 0xFFFF {
                    break;
                }

                if fields as usize != cols {
                    // Skip invalid row
                    for _ in 0..fields {
                        let len = reader.read_u32::<BE>().unwrap_or(0);
                        reader
                            .by_ref()
                            .take(len as u64)
                            .read_to_end(&mut Vec::new())
                            .ok();
                    }
                    continue;
                }

                let mut row: Vec<Field> = Vec::with_capacity(cols);
                for ty in T::columns() {
                    let len = reader.read_u32::<BE>().unwrap();
                    let _ = len;
                    match *ty {
                        tokio_postgres::types::Type::INT8 => {
                            let val = reader.read_i64::<BE>().unwrap();
                            row.push(Field::I64(val));
                        }
                        tokio_postgres::types::Type::FLOAT4 => {
                            let val = reader.read_f32::<BE>().unwrap();
                            row.push(Field::F32(val));
                        }
                        _ => unreachable!("unsupported column type"),
                    }
                }

                buffer.push(row);

                if buffer.len() == BATCH {
                    for row in &buffer {
                        writer.as_mut().write_raw(row.iter()).await?;
                    }
                    buffer.clear();

                    batch_count += 1;
                    if batch_count % 100 == 0 {
                        log::info!("COPY progress: {} batches processed", batch_count);
                    }
                }
            }
        }

        // Write remaining rows
        for row in &buffer {
            writer.as_mut().write_raw(row.iter()).await?;
        }

        writer.finish().await?;
        Ok(())
    }

    async fn vacant(&self, table: &str) -> Result<bool, E> {
        let ref sql = format!(
            "
            SELECT 1
            FROM   {}
            LIMIT  1;
            ",
            table
        );
        Ok(self.0.query(sql, &[]).await?.is_empty())
    }

    async fn absent(&self, table: &str) -> Result<bool, E> {
        let ref sql = format!(
            "
            SELECT  1
            FROM    information_schema.tables
            WHERE   table_name = '{}';
            ",
            table
        );
        Ok(self.0.query(sql, &[]).await?.is_empty())
    }
}

/// Zero-copy field representation for PostgreSQL binary format
#[derive(Debug)]
enum Field {
    F32(f32),
    I64(i64),
}

impl tokio_postgres::types::ToSql for Field {
    fn to_sql(
        &self,
        ty: &tokio_postgres::types::Type,
        out: &mut bytes::BytesMut,
    ) -> Result<tokio_postgres::types::IsNull, Box<dyn std::error::Error + Sync + Send>> {
        match self {
            Field::F32(val) => val.to_sql(ty, out),
            Field::I64(val) => val.to_sql(ty, out),
        }
    }

    fn accepts(ty: &tokio_postgres::types::Type) -> bool {
        <f32 as ToSql>::accepts(ty) || <i64 as ToSql>::accepts(ty)
    }

    fn to_sql_checked(
        &self,
        ty: &tokio_postgres::types::Type,
        out: &mut bytes::BytesMut,
    ) -> Result<tokio_postgres::types::IsNull, Box<dyn std::error::Error + Sync + Send>> {
        match self {
            Field::F32(val) => val.to_sql_checked(ty, out),
            Field::I64(val) => val.to_sql_checked(ty, out),
        }
    }
}
