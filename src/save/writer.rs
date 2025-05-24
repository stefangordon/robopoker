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
use futures::pin_mut;
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

        // Speed up large COPY operations for this session.
        // – no fsync for each statement
        // – larger work_mem buffers for index build later
        postgres
            .0
            .batch_execute(
                "SET synchronous_commit = OFF; SET temp_buffers = '512MB';",
            )
            .await?;

        postgres.upload::<Metric>().await?;
        postgres.upload::<Decomp>().await?;
        postgres.upload::<Encoder>().await?;
        postgres.upload::<Profile>().await?;
        postgres.derive::<Abstraction>().await?;
        postgres.derive::<Street>().await?;
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
            self.stream::<T>().await?;
            self.0.batch_execute(&T::indices()).await?;
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

    async fn stream<T>(&self) -> Result<(), E>
    where
        T: Table,
    {
        let sink = self.0.copy_in(&T::copy()).await?;
        let writer = BinaryCopyInWriter::new(sink, T::columns());
        pin_mut!(writer);

        // Tune batch size for best throughput.
        const BATCH: usize = 8_192;
        let cols = T::columns().len();
        let mut buffer: Vec<Vec<Field>> = Vec::with_capacity(BATCH);

        // Simple progress tracking for batches (works for both compressed and uncompressed)
        let mut batch_count: usize = 0;

        let mut tag = [0u8; 2];

        for src in T::sources() {
            // Detect zstd compression by checking magic bytes
            let mut magic = [0u8; 4];
            let mut file_for_magic = File::open(&src).expect("file not found");
            let _ = file_for_magic.read_exact(&mut magic);
            file_for_magic.seek(SeekFrom::Start(0)).unwrap();
            
            // Create appropriate reader based on compression detection
            let reader_inner: Box<dyn Read> = if magic == [0x28, 0xB5, 0x2F, 0xFD] {
                log::debug!("Detected zstd compression in file: {}", src);
                Box::new(ZDecoder::new(file_for_magic).expect("zstd decode"))
            } else {
                log::debug!("Using uncompressed file: {}", src);
                Box::new(file_for_magic)
            };
            
            let mut reader = BufReader::new(reader_inner);
            
            // Skip the 19-byte PostgreSQL binary header
            // For zstd streams, we read and discard since they don't support seeking
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
                    // Unexpected field count – skip this row gracefully.
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
                    // rows are sent immediately; buffer cleared to keep memory bounded
                    buffer.clear();
                    
                    // Progress tracking
                    batch_count += 1;
                    if batch_count % 100 == 0 {
                        log::info!("COPY progress: {} batches processed", batch_count);
                    }
                }
            }
        }

        // write any remaining rows
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

/// doing this for zero copy reasons
/// it was impossible to achieve polymorphism between column types
/// without allocating a ton since writer.as_mut().write()
/// required &[&dyn ToSql]
/// which would have required collection into a Vec<&dyn ToSql>
/// because of lifetime reasons. now, we only need an Iterator<Item = T: ToSql>
/// which is much more flexible, so we can map T::columns() to dynamically
/// iterate over table columns.
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
