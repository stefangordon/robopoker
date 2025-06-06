use super::edge::Edge;
use super::game::Game;
use super::info::Info;
use super::turn::Turn;
use crate::cards::isomorphism::Isomorphism;
use crate::cards::street::Street;
use crate::clustering::abstraction::Abstraction;
use crate::clustering::Lookup;
use crate::gameplay::action::Action;
use crate::gameplay::odds::Odds;
use crate::gameplay::path::Path;
use crate::mccfr::types::branch::Branch;
use std::collections::BTreeMap;
use std::marker::PhantomData;

type Tree = crate::mccfr::structs::tree::Tree<Turn, Edge, Game, Info>;

/// Strategy object that determines which bet sizes are allowed at a given
/// game state and depth.  This lets us reuse the full NLHE encoder for both
/// blueprint training (coarse grid) and sub-game re-solving (rich grid)
/// simply by swapping the type parameter.
pub trait BetSizer {
    fn raises(game: &Game, depth: usize) -> Vec<Odds>;
}

/// Original blueprint sizing behaviour – identical code that used to live
/// inside `Encoder::raises`.
pub struct BlueprintSizer;
impl BetSizer for BlueprintSizer {
    fn raises(game: &Game, depth: usize) -> Vec<Odds> {
        if depth > crate::MAX_RAISE_REPEATS {
            vec![]
        } else {
            match game.street() {
                Street::Pref => Odds::PREF_RAISES.to_vec(),
                Street::Flop => Odds::FLOP_RAISES.to_vec(),
                _ => match depth {
                    0 => Odds::LATE_RAISES.to_vec(),
                    _ => Odds::LAST_RAISES.to_vec(),
                },
            }
        }
    }
}

pub type BlueprintEncoder = Encoder<BlueprintSizer>;

#[derive(Default)]
pub struct Encoder<S: BetSizer = BlueprintSizer> {
    lookup: BTreeMap<Isomorphism, Abstraction>,
    _phantom: PhantomData<S>,
}

impl<S: BetSizer> Encoder<S> {
    fn name() -> String {
        format!("isomorphism_{}", std::any::type_name::<S>())
    }

    pub fn abstraction(&self, iso: &Isomorphism) -> Abstraction {
        // We call this hundreds of thousands of times during roll-outs, so avoid
        // the `format!` allocation that happens even on success. If an entry is
        // ever missing we panic with a concise message.
        self.lookup.get(iso).copied().unwrap_or_else(|| {
            panic!("Missing abstraction for isomorphism – lookup table incomplete")
        })
    }

    /// Get a clone of the abstraction lookup
    pub fn get_lookup(&self) -> BTreeMap<Isomorphism, Abstraction> {
        self.lookup.clone()
    }

    /// Construct an Encoder with a pre-loaded abstraction lookup (used by subgame re-solver)
    pub fn new(lookup: BTreeMap<Isomorphism, Abstraction>) -> Self {
        Self {
            lookup,
            _phantom: PhantomData,
        }
    }

    pub fn choices(game: &Game, depth: usize) -> Vec<Edge> {
        game.legal()
            .into_iter()
            .flat_map(|action| Self::unfold(game, depth, action))
            .collect()
    }

    pub fn raises(game: &Game, depth: usize) -> Vec<Odds> {
        S::raises(game, depth)
    }

    pub fn unfold(game: &Game, depth: usize, action: Action) -> Vec<Edge> {
        match action {
            Action::Raise(_) => Self::raises(game, depth)
                .into_iter()
                .map(Edge::from)
                .collect::<Vec<Edge>>(),
            _ => vec![Edge::from(action)],
        }
    }

    #[allow(dead_code)]
    fn infoize(&self, recall: &crate::gameplay::recall::Recall) -> Info {
        let depth = 0;
        let ref game = recall.head();
        let ref iso = recall.isomorphism();
        let present = self.abstraction(iso);
        let futures = Path::from(Self::choices(game, depth));
        let history = Path::from(recall.path());
        Info::from((history, present, futures))
    }
}

impl<S: BetSizer> crate::mccfr::traits::encoder::Encoder for Encoder<S> {
    type T = Turn;
    type E = Edge;
    type G = Game;
    type I = Info;
    type S = S;

    fn seed(&self, root: &Self::G) -> Self::I {
        let ref iso = Isomorphism::from(root.sweat());
        let depth = 0;
        let present = self.abstraction(iso);
        let history = Path::default();
        let futures = Path::from(Self::choices(root, depth));
        Self::I::from((history, present, futures))
    }

    fn info(&self, tree: &Tree, leaf: Branch<Self::E, Self::G>) -> Self::I {
        let (edge, ref game, head) = leaf;
        let ref iso = Isomorphism::from(game.sweat());
        let n_raises = tree
            .at(head)
            .into_iter()
            .take_while(|(_, e)| e.is_choice())
            .filter(|(_, e)| e.is_aggro())
            .count();
        let present = self.abstraction(iso);
        let futures = Path::from(Self::choices(game, n_raises));
        let history = std::iter::once(edge)
            .chain(tree.at(head).into_iter().map(|(_, e)| e))
            .take(crate::MAX_DEPTH_SUBGAME)
            .collect::<Path>();
        Self::I::from((history, present, futures))
    }
}

#[cfg(feature = "native")]
impl<S: BetSizer> crate::save::upload::Table for Encoder<S> {
    fn name() -> String {
        Self::name()
    }
    fn columns() -> &'static [tokio_postgres::types::Type] {
        &[
            tokio_postgres::types::Type::INT8,
            tokio_postgres::types::Type::INT8,
        ]
    }
    fn sources() -> Vec<String> {
        use crate::save::disk::Disk;
        Street::all()
            .iter()
            .rev()
            .copied()
            .map(Lookup::path)
            .collect()
    }
    fn creates() -> String {
        "
            CREATE TABLE IF NOT EXISTS isomorphism (
                obs        BIGINT,
                abs        BIGINT,
                position   INTEGER
            );"
        .to_string()
    }
    fn indices() -> String {
        "
            CREATE INDEX IF NOT EXISTS idx_isomorphism_covering     ON isomorphism  (obs, abs) INCLUDE (abs);
            CREATE INDEX IF NOT EXISTS idx_isomorphism_abs_position ON isomorphism  (abs, position);
            CREATE INDEX IF NOT EXISTS idx_isomorphism_abs_obs      ON isomorphism  (abs, obs);
            CREATE INDEX IF NOT EXISTS idx_isomorphism_abs          ON isomorphism  (abs);
            CREATE INDEX IF NOT EXISTS idx_isomorphism_obs          ON isomorphism  (obs);
            --
            WITH numbered AS (
                SELECT obs, abs, row_number() OVER (PARTITION BY abs ORDER BY obs) - 1 as rn
                FROM isomorphism
            )
                UPDATE isomorphism
                SET    position = numbered.rn
                FROM   numbered
                WHERE  isomorphism.obs = numbered.obs
                AND    isomorphism.abs = numbered.abs;
            "
            .to_string()
    }
    fn copy() -> String {
        "
            COPY isomorphism (
                obs,
                abs
            )
            FROM STDIN BINARY
            "
        .to_string()
    }
}

impl<S: BetSizer> crate::save::disk::Disk for Encoder<S> {
    fn name() -> String {
        "isomorphism".to_string()
    }
    fn path(street: Street) -> String {
        format!(
            "{}/pgcopy/isomorphism.{}",
            std::env::current_dir()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned(),
            street
        )
    }
    fn done(street: Street) -> bool {
        let path = Self::path(street);
        std::fs::metadata(path).is_ok()
    }
    fn save(&self) {
        unimplemented!("saving happens at Lookup level. composed of 4 street-level Lookup saves")
    }
    fn grow(_: Street) -> Self {
        unimplemented!("you have no business making an encoding from scratch, learn from kmeans")
    }
    fn load(_: Street) -> Self {
        Self {
            lookup: Street::all()
                .iter()
                .copied()
                .map(crate::clustering::lookup::Lookup::load)
                .map(BTreeMap::from)
                .fold(BTreeMap::default(), |mut map, l| {
                    map.extend(l);
                    map
                }),
            _phantom: PhantomData,
        }
    }
}
