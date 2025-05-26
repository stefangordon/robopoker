use crate::cards::isomorphism::Isomorphism;
use crate::cards::isomorphisms::IsomorphismIterator;
use crate::cards::observation::Observation;
use crate::cards::street::Street;
use crate::clustering::abstraction::Abstraction;
use crate::clustering::histogram::Histogram;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::OnceLock;
use crate::save::disk::Disk;

#[derive(Default)]
/// this is the grand lookup table for all the Isomorphism -> Abstraction
/// mappings. we spend a lot of compute over a lot of hands (all of them!)
/// to precompute this mapping.
pub struct Lookup(BTreeMap<Isomorphism, Abstraction>);

impl From<Lookup> for BTreeMap<Isomorphism, Abstraction> {
    fn from(lookup: Lookup) -> BTreeMap<Isomorphism, Abstraction> {
        lookup.0
    }
}
impl From<BTreeMap<Isomorphism, Abstraction>> for Lookup {
    fn from(map: BTreeMap<Isomorphism, Abstraction>) -> Self {
        Self(map)
    }
}

impl Lookup {
    /// lookup the pre-computed abstraction for the outer observation
    pub fn lookup(&self, obs: &Observation) -> Abstraction {
        self.0
            .get(&Isomorphism::from(*obs))
            .cloned()
            .expect(&format!("precomputed abstraction missing for {obs}"))
    }
    #[cfg(feature = "native")]
    /// generate the entire space of inner layers
    pub fn projections(&self) -> Vec<Histogram> {
        use rayon::iter::IntoParallelIterator;
        use rayon::iter::ParallelIterator;
        IsomorphismIterator::from(self.street().prev())
            .collect::<Vec<Isomorphism>>()
            .into_par_iter()
            .map(|inner| self.future(&inner))
            .collect::<Vec<Histogram>>()
    }
    /// distribution over potential next states. this "layer locality" is what
    /// makes imperfect recall hierarchical kmeans nice
    fn future(&self, iso: &Isomorphism) -> Histogram {
        assert!(iso.0.street() != Street::Rive);
        iso.0
            .children()
            .map(|o| self.lookup(&o))
            .collect::<Vec<Abstraction>>()
            .into()
    }
    fn street(&self) -> Street {
        self.0.keys().next().expect("non empty").0.street()
    }
    fn name() -> String {
        "isomorphism".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[ignore]
    #[test]
    fn persistence() {
        let street = Street::Pref;
        let lookup = Lookup::grow(street);
        lookup.save();
        let loaded = Lookup::load(street);
        std::iter::empty()
            .chain(lookup.0.iter().zip(loaded.0.iter()))
            .chain(loaded.0.iter().zip(lookup.0.iter()))
            .all(|((s1, l1), (s2, l2))| s1 == s2 && l1 == l2);
    }
}

impl crate::save::disk::Disk for Lookup {
    fn name() -> String {
        Self::name()
    }
    /// abstractions for River are calculated once via obs.equity
    /// abstractions for Preflop are cequivalent to just enumerating isomorphisms
    fn grow(street: Street) -> Self {
        use rayon::iter::IntoParallelIterator;
        use rayon::iter::ParallelIterator;
        match street {
            Street::Rive => IsomorphismIterator::from(Street::Rive)
                .collect::<Vec<_>>()
                .into_par_iter()
                .map(|iso| (iso, Abstraction::from(iso.0.equity())))
                .collect::<BTreeMap<_, _>>()
                .into(),
            Street::Pref => IsomorphismIterator::from(Street::Pref)
                .enumerate()
                .map(|(k, iso)| (iso, Abstraction::from((Street::Pref, k))))
                .collect::<BTreeMap<_, _>>()
                .into(),
            Street::Flop | Street::Turn => panic!("lookup must be learned via layer for {street}"),
        }
    }
    fn load(street: Street) -> Self {
        let ref path = Self::path(street);
        log::info!("{:<32}{:<32}", "loading     lookup", path);
        use byteorder::{BE, ReadBytesExt};
        use std::fs::File;
        use std::io::BufReader;
        use std::io::Read;
        use std::io::Seek;
        use std::io::SeekFrom;
        let ref file = File::open(path).expect(&format!("open {}", path));
        let mut lookup = BTreeMap::new();
        let mut reader = BufReader::with_capacity(4 * 1024 * 1024, file);
        let ref mut buffer = [0u8; 2];
        reader.seek(SeekFrom::Start(19)).expect("seek past header");
        while reader.read_exact(buffer).is_ok() {
            match u16::from_be_bytes(buffer.clone()) {
                2 => {
                    assert!(8 == reader.read_u32::<BE>().expect("observation length"));
                    let iso = reader.read_i64::<BE>().expect("read observation");
                    assert!(8 == reader.read_u32::<BE>().expect("abstraction length"));
                    let abs = reader.read_i64::<BE>().expect("read abstraction");
                    let observation = Isomorphism::from(iso);
                    let abstraction = Abstraction::from(abs);
                    lookup.insert(observation, abstraction);
                }
                0xFFFF => break,
                n => panic!("unexpected number of fields: {}", n),
            }
        }
        Self(lookup)
    }
    fn save(&self) {
        const N_FIELDS: u16 = 2;
        let street = self.street();
        let ref path = Self::path(street);
        let ref mut file = File::create(path).expect(&format!("touch {}", path));
        use byteorder::{BE, WriteBytesExt};
        use std::fs::File;
        use std::io::Write;
        use std::io::BufWriter;
        use std::mem::size_of;
        // Use a large buffer (4 MB) for faster writes
        const BUF: usize = 4 * 1024 * 1024;

        let mut writer = BufWriter::with_capacity(BUF, file);
        log::info!("{:<32}{:<32}", "saving      lookup", path);
        writer.write_all(Self::header()).expect("header");
        for (Isomorphism(obs), abs) in self.0.iter() {
            writer.write_u16::<BE>(N_FIELDS).unwrap();
            writer.write_u32::<BE>(size_of::<i64>() as u32).unwrap();
            writer.write_i64::<BE>(i64::from(*obs)).unwrap();
            writer.write_u32::<BE>(size_of::<i64>() as u32).unwrap();
            writer.write_i64::<BE>(i64::from(*abs)).unwrap();
        }
        writer.write_u16::<BE>(Self::footer()).expect("trailer");
        writer.flush().expect("flush writer");
    }
}

// Static caches so large lookups are loaded once per process.
static PREF_LOOKUP: OnceLock<Arc<BTreeMap<Isomorphism, Abstraction>>> = OnceLock::new();
static FLOP_LOOKUP: OnceLock<Arc<BTreeMap<Isomorphism, Abstraction>>> = OnceLock::new();
static TURN_LOOKUP: OnceLock<Arc<BTreeMap<Isomorphism, Abstraction>>> = OnceLock::new();
static RIVE_LOOKUP: OnceLock<Arc<BTreeMap<Isomorphism, Abstraction>>> = OnceLock::new();

fn init_lookup(cell: &OnceLock<Arc<BTreeMap<Isomorphism, Abstraction>>>, street: Street) -> Arc<BTreeMap<Isomorphism, Abstraction>> {
    cell.get_or_init(|| {
        log::info!("loading abstraction lookup for {:?}", street);
        let lookup: Lookup = <Lookup as Disk>::load(street);
        Arc::new(lookup.into())
    }).clone()
}

/// Retrieve a cached abstraction lookup for the given street.  Heavy files are
/// loaded exactly once (on first access) and then shared via `Arc`.
pub fn cached(street: Street) -> Arc<BTreeMap<Isomorphism, Abstraction>> {
    match street {
        Street::Pref => init_lookup(&PREF_LOOKUP, Street::Pref),
        Street::Flop => init_lookup(&FLOP_LOOKUP, Street::Flop),
        Street::Turn => init_lookup(&TURN_LOOKUP, Street::Turn),
        Street::Rive => init_lookup(&RIVE_LOOKUP, Street::Rive),
    }
}
