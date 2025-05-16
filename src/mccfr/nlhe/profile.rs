use super::edge::Edge;
use super::game::Game;
use super::info::Info;
use super::turn::Turn;
use crate::cards::street::Street;
use std::collections::BTreeMap;
use std::io::Write;
use std::io::BufWriter;
use std::fs::File;

#[derive(Default)]
pub struct Profile {
    pub(super) iterations: usize,
    pub(super) encounters: BTreeMap<Info, BTreeMap<Edge, (crate::Probability, crate::Utility)>>,
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
            .and_then(|memory| memory.get(edge))
            .map(|(w, _)| *w)
            .unwrap_or_default()
    }
    fn sum_regret(&self, info: &Self::I, edge: &Self::E) -> crate::Utility {
        self.encounters
            .get(info)
            .and_then(|memory| memory.get(edge))
            .map(|(_, r)| *r)
            .unwrap_or_default()
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
        format!(
            "{}/pgcopy/{}",
            std::env::current_dir()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned(),
            Self::name()
        )
    }
    fn load(_: Street) -> Self {
        let ref path = Self::path(Street::random());
        log::info!("{:<32}{:<32}", "loading     blueprint", path);
        use crate::clustering::abstraction::Abstraction;
        use crate::gameplay::path::Path;
        use crate::mccfr::nlhe::info::Info;
        use crate::Arbitrary;
        use byteorder::ReadBytesExt;
        use byteorder::BE;
        use std::fs::File;
        use std::io::BufReader;
        use std::io::Read;
        use std::io::Seek;
        use std::io::SeekFrom;
        let file = File::open(path).expect("open file");
        let mut encounters = BTreeMap::new();
        let mut reader = BufReader::new(file);
        let ref mut buffer = [0u8; 2];
        reader.seek(SeekFrom::Start(19)).expect("seek past header");
        while reader.read_exact(buffer).is_ok() {
            match u16::from_be_bytes(buffer.clone()) {
                6 => {
                    reader.read_u32::<BE>().expect("past path length");
                    let history = Path::from(reader.read_u64::<BE>().expect("history"));
                    reader.read_u32::<BE>().expect("abstraction length");
                    let present = Abstraction::from(reader.read_u64::<BE>().expect("abstraction"));
                    reader.read_u32::<BE>().expect("future path length");
                    let choices = Path::from(reader.read_u64::<BE>().expect("choices"));
                    reader.read_u32::<BE>().expect("edge length");
                    let edge = Edge::from(reader.read_u64::<BE>().expect("read edge"));
                    reader.read_u32::<BE>().expect("regret length");
                    let regret = reader.read_f32::<BE>().expect("read regret");
                    reader.read_u32::<BE>().expect("policy length");
                    let policy = reader.read_f32::<BE>().expect("read policy");
                    let bucket = Info::from((history, present, choices));
                    encounters
                        .entry(bucket)
                        .or_insert_with(BTreeMap::default)
                        .entry(edge)
                        .or_insert((policy, regret));
                }
                0xFFFF => break,
                n => panic!("unexpected number of fields: {}", n),
            }
        }
        Self {
            encounters,
            iterations: 0,
        }
    }
    fn save(&self) {
        const N_FIELDS: u16 = 6;
        let ref path = Self::path(Street::random());
        let file = File::create(path).expect(&format!("touch {}", path));
        let mut writer = BufWriter::with_capacity(1024 * 1024, file);
        let total_buckets = self.encounters.len();
        log::info!("Saving blueprint to {} ({} info-sets)", path, total_buckets);
        use crate::Arbitrary;
        use byteorder::WriteBytesExt;
        use byteorder::BE;
        log::info!("{:<32}{:<32}", "saving      blueprint", path);
        writer.write_all(Self::header()).expect("header");
        for (idx, (_bucket, strategy)) in self.encounters.iter().enumerate() {
            if idx % 50000 == 0 || idx + 1 == total_buckets {
                log::info!("  saved {}/{} info-sets", idx + 1, total_buckets);
            }
            for (edge, memory) in strategy.iter() {
                writer.write_u16::<BE>(N_FIELDS).unwrap();
                writer.write_u32::<BE>(size_of::<u64>() as u32).unwrap();
                writer.write_u64::<BE>(u64::from(*_bucket.history())).unwrap();
                writer.write_u32::<BE>(size_of::<u64>() as u32).unwrap();
                writer.write_u64::<BE>(u64::from(*_bucket.present())).unwrap();
                writer.write_u32::<BE>(size_of::<u64>() as u32).unwrap();
                writer.write_u64::<BE>(u64::from(*_bucket.futures())).unwrap();
                writer.write_u32::<BE>(size_of::<u64>() as u32).unwrap();
                writer.write_u64::<BE>(u64::from(edge.clone())).unwrap();
                writer.write_u32::<BE>(size_of::<f32>() as u32).unwrap();
                writer.write_f32::<BE>(memory.1).unwrap();
                writer.write_u32::<BE>(size_of::<f32>() as u32).unwrap();
                writer.write_f32::<BE>(memory.0).unwrap();
            }
        }
        writer.write_u16::<BE>(Self::footer()).expect("trailer");
    }
}
