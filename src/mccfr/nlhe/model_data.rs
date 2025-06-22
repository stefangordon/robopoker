use super::encoder::BlueprintEncoder;
use super::{Edge, Game, Info, Profile, Turn};
use crate::cards::isomorphism::Isomorphism;
use crate::cards::street::Street;
use crate::gameplay::path::Path;
use crate::gameplay::seat::State;
use crate::mccfr::traits::profile::Profile as ProfileTrait;
use arrow2::{
    array::*,
    chunk::Chunk,
    datatypes::{DataType, Field, Schema},
    io::parquet::write::*,
};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path as FilePath, PathBuf};

/// Size of chunks for writing parquet data
const CHUNK_SIZE: usize = 10_000;

/// Output directory for model training data
const OUTPUT_DIR: &str = "modeldata";

/// Output filename for training data
const OUTPUT_FILE: &str = "training_data.parquet";

impl Profile {
    /// Generate model training data and write to parquet file
    pub fn generate_model_data(
        &self,
        encoder: &BlueprintEncoder,
    ) -> Result<(), arrow2::error::Error> {
        log::info!("Starting model data generation from profile");

        let output_path = prepare_output_path()?;
        let schema = create_model_schema();
        let mut writer = create_parquet_writer(&output_path, schema.clone())?;
        let mut buffer = ChunkBuffer::new();

        let progress = crate::progress(self.encounters.len());
        progress.set_message("Generating model training data");

        let mut processed = 0;
        for entry in self.encounters.iter() {
            if let Some(record) = process_infoset(self, encoder, entry) {
                buffer.add(record);
                processed += 1;

                if buffer.is_full() {
                    buffer.write_chunk(&mut writer, &schema)?;
                }

                if processed % 1000 == 0 {
                    progress.set_position(processed as u64);
                }
            }
        }

        // Write any remaining data
        if !buffer.is_empty() {
            buffer.write_chunk(&mut writer, &schema)?;
        }

        writer.end(None)?;
        progress.finish_with_message("Model data generation complete");

        log::info!("Model data saved to: {}", output_path.display());
        log::info!("Total records written: {}", processed);

        Ok(())
    }
}

/// Prepare output directory and return path
fn prepare_output_path() -> Result<PathBuf, std::io::Error> {
    let output_dir = FilePath::new(OUTPUT_DIR);
    std::fs::create_dir_all(output_dir)?;
    Ok(output_dir.join(OUTPUT_FILE))
}

/// Create parquet writer with appropriate settings
fn create_parquet_writer(
    path: &FilePath,
    schema: Schema,
) -> Result<FileWriter<std::fs::File>, arrow2::error::Error> {
    let options = WriteOptions {
        write_statistics: false,
        compression: CompressionOptions::Zstd(Some(ZstdLevel::try_new(1)?)),
        version: Version::V2,
        data_pagesize_limit: None,
    };

    let file = std::fs::File::create(path)?;
    FileWriter::try_new(file, schema, options)
}

/// Process a single infoset and extract all features
fn process_infoset(
    profile: &Profile,
    encoder: &BlueprintEncoder,
    entry: dashmap::mapref::multiple::RefMulti<
        '_,
        Info,
        parking_lot::RwLock<super::compact_bucket::CompactBucket>,
    >,
) -> Option<ModelRecord> {
    let info = entry.key();
    let bucket = entry.value().read();

    if bucket.is_empty() {
        return None;
    }

    let game = replay_to_infoset(&Game::root(), info);
    let features = extract_game_features(&game, info);

    Some(ModelRecord {
        infoset_id: compute_infoset_id(info),
        reach_prob: calculate_reach_prob(profile, info, encoder),
        game_features: features,
        cluster: u64::from(*info.present()) as u16,
        action_seq: extract_action_sequence(info, &game),
        legal_actions: extract_legal_actions(info),
        advantages: extract_advantages(info, &bucket),
    })
}

/// Compute unique identifier for an infoset
fn compute_infoset_id(info: &Info) -> i64 {
    let mut hasher = DefaultHasher::new();
    info.hash(&mut hasher);
    hasher.finish() as i64
}

/// Extract legal actions from infoset
fn extract_legal_actions(info: &Info) -> Vec<u16> {
    info.futures()
        .clone()
        .map(|edge| u8::from(edge) as u16)
        .collect()
}

/// Extract advantages (positive regrets) for each action
fn extract_advantages(
    info: &Info,
    bucket: &parking_lot::lock_api::RwLockReadGuard<
        '_,
        parking_lot::RawRwLock,
        super::compact_bucket::CompactBucket,
    >,
) -> Vec<f32> {
    info.futures()
        .clone()
        .map(|edge| {
            let edge_u8 = u8::from(edge);
            bucket
                .iter()
                .find(|(e, _)| *e == edge_u8)
                .map(|(_, (_, regret))| regret.max(0.0))
                .unwrap_or(0.0)
        })
        .collect()
}

/// Game state features
struct GameFeatures {
    n_players: u8,
    position: u8,
    street: u8,
    eff_stack_bb: f32,
    pot_bb: f32,
    to_call_bb: f32,
    bet_frac_pot: Vec<f32>,
    spr: f32,
}

/// Complete record for one training example
struct ModelRecord {
    infoset_id: i64,
    reach_prob: f32,
    game_features: GameFeatures,
    cluster: u16,
    action_seq: Vec<u8>,
    legal_actions: Vec<u16>,
    advantages: Vec<f32>,
}

/// Buffer for accumulating chunks of data
struct ChunkBuffer {
    infoset_ids: Vec<i64>,
    reach_probs: Vec<f32>,
    n_players: Vec<u8>,
    positions: Vec<u8>,
    streets: Vec<u8>,
    clusters: Vec<u16>,
    eff_stacks: Vec<f32>,
    pots: Vec<f32>,
    to_calls: Vec<f32>,
    bet_fracs: Vec<Vec<f32>>,
    sprs: Vec<f32>,
    action_seqs: Vec<Vec<u8>>,
    legal_actions: Vec<Vec<u16>>,
    advantages: Vec<Vec<f32>>,
}

impl ChunkBuffer {
    fn new() -> Self {
        Self {
            infoset_ids: Vec::with_capacity(CHUNK_SIZE),
            reach_probs: Vec::with_capacity(CHUNK_SIZE),
            n_players: Vec::with_capacity(CHUNK_SIZE),
            positions: Vec::with_capacity(CHUNK_SIZE),
            streets: Vec::with_capacity(CHUNK_SIZE),
            clusters: Vec::with_capacity(CHUNK_SIZE),
            eff_stacks: Vec::with_capacity(CHUNK_SIZE),
            pots: Vec::with_capacity(CHUNK_SIZE),
            to_calls: Vec::with_capacity(CHUNK_SIZE),
            bet_fracs: Vec::with_capacity(CHUNK_SIZE),
            sprs: Vec::with_capacity(CHUNK_SIZE),
            action_seqs: Vec::with_capacity(CHUNK_SIZE),
            legal_actions: Vec::with_capacity(CHUNK_SIZE),
            advantages: Vec::with_capacity(CHUNK_SIZE),
        }
    }

    fn add(&mut self, record: ModelRecord) {
        self.infoset_ids.push(record.infoset_id);
        self.reach_probs.push(record.reach_prob);
        self.n_players.push(record.game_features.n_players);
        self.positions.push(record.game_features.position);
        self.streets.push(record.game_features.street);
        self.clusters.push(record.cluster);
        self.eff_stacks.push(record.game_features.eff_stack_bb);
        self.pots.push(record.game_features.pot_bb);
        self.to_calls.push(record.game_features.to_call_bb);
        self.bet_fracs.push(record.game_features.bet_frac_pot);
        self.sprs.push(record.game_features.spr);
        self.action_seqs.push(record.action_seq);
        self.legal_actions.push(record.legal_actions);
        self.advantages.push(record.advantages);
    }

    fn is_full(&self) -> bool {
        self.infoset_ids.len() >= CHUNK_SIZE
    }

    fn is_empty(&self) -> bool {
        self.infoset_ids.is_empty()
    }

    fn write_chunk(
        &mut self,
        writer: &mut FileWriter<std::fs::File>,
        schema: &Schema,
    ) -> Result<(), arrow2::error::Error> {
        let arrays: Vec<Box<dyn Array>> = vec![
            Box::new(Int64Array::from_slice(&self.infoset_ids)),
            Box::new(Float32Array::from_slice(&self.reach_probs)),
            Box::new(UInt8Array::from_slice(&self.n_players)),
            Box::new(UInt8Array::from_slice(&self.positions)),
            Box::new(UInt8Array::from_slice(&self.streets)),
            Box::new(UInt16Array::from_slice(&self.clusters)),
            Box::new(Float32Array::from_slice(&self.eff_stacks)),
            Box::new(Float32Array::from_slice(&self.pots)),
            Box::new(Float32Array::from_slice(&self.to_calls)),
            create_list_array::<f32, MutablePrimitiveArray<f32>>(&self.bet_fracs),
            Box::new(Float32Array::from_slice(&self.sprs)),
            create_list_array::<u8, MutablePrimitiveArray<u8>>(&self.action_seqs),
            create_list_array::<u16, MutablePrimitiveArray<u16>>(&self.legal_actions),
            create_list_array::<f32, MutablePrimitiveArray<f32>>(&self.advantages),
        ];

        let chunk = Chunk::new(arrays);
        let encodings = vec![vec![Encoding::Plain; schema.fields.len()]];

        let row_groups = RowGroupIterator::try_new(
            vec![Ok(chunk)].into_iter(),
            schema,
            writer.options(),
            encodings,
        )?;

        for group in row_groups {
            writer.write(group?)?;
        }

        self.clear();
        Ok(())
    }

    fn clear(&mut self) {
        self.infoset_ids.clear();
        self.reach_probs.clear();
        self.n_players.clear();
        self.positions.clear();
        self.streets.clear();
        self.clusters.clear();
        self.eff_stacks.clear();
        self.pots.clear();
        self.to_calls.clear();
        self.bet_fracs.clear();
        self.sprs.clear();
        self.action_seqs.clear();
        self.legal_actions.clear();
        self.advantages.clear();
    }
}

/// Create the schema for model training data
fn create_model_schema() -> Schema {
    use DataType::*;

    let list_f32 = List(Box::new(Field::new("item", Float32, true)));
    let list_u8 = List(Box::new(Field::new("item", UInt8, true)));
    let list_u16 = List(Box::new(Field::new("item", UInt16, true)));

    Schema::from(vec![
        Field::new("infoset_id", Int64, false),
        Field::new("reach_prob", Float32, false),
        Field::new("n_players", UInt8, false),
        Field::new("position", UInt8, false),
        Field::new("street", UInt8, false),
        Field::new("cluster", UInt16, false),
        Field::new("eff_stack_bb", Float32, false),
        Field::new("pot_bb", Float32, false),
        Field::new("to_call_bb", Float32, false),
        Field::new("bet_frac_pot", list_f32.clone(), false),
        Field::new("spr", Float32, false),
        Field::new("action_seq", list_u8, false),
        Field::new("legal_actions", list_u16, false),
        Field::new("adv_plus", list_f32, false),
    ])
}

/// Calculate reach probability by replaying the path
fn calculate_reach_prob(profile: &Profile, target_info: &Info, encoder: &BlueprintEncoder) -> f32 {
    let mut reach_prob = 1.0;
    let mut game = Game::root();
    let mut path_so_far = Vec::new();
    let mut n_raises = 0;

    for edge in target_info.history().clone() {
        if let Turn::Choice(_) = game.turn() {
            let current_info = build_current_info(&game, encoder, &path_so_far, n_raises);
            reach_prob *= profile.advice(&current_info, &edge);
        }

        if edge.is_aggro() {
            n_raises += 1;
        }

        game = game.apply(game.actionize(&edge));
        path_so_far.push(edge);

        if matches!(game.turn(), Turn::Terminal) {
            break;
        }
    }

    reach_prob
}

/// Build information set for current game state
fn build_current_info(
    game: &Game,
    encoder: &BlueprintEncoder,
    path_so_far: &[Edge],
    n_raises: usize,
) -> Info {
    let iso = Isomorphism::from(game.sweat());
    let abstraction = encoder.abstraction(&iso);
    let choices = BlueprintEncoder::choices(game, n_raises);
    let futures = Path::from(choices);
    let partial_history = Path::from(path_so_far.to_vec());
    Info::from((partial_history, abstraction, futures))
}

/// Replay game to reach the given infoset
fn replay_to_infoset(initial_game: &Game, info: &Info) -> Game {
    info.history()
        .clone()
        .fold(initial_game.clone(), |game, edge| {
            game.apply(game.actionize(&edge))
        })
}

/// Extract game features from the current game state
fn extract_game_features(game: &Game, info: &Info) -> GameFeatures {
    let n_players = count_active_players(game);
    let hero_idx = get_hero_index(game);
    let position = calculate_position(game, hero_idx);
    let street = encode_street(game.street());

    let bb = crate::B_BLIND as f32;
    let pot_bb = game.pot() as f32 / bb;
    let to_call_bb = game.to_call() as f32 / bb;
    let eff_stack_bb = game.effective_stack() as f32 / bb;

    let bet_frac_pot = calculate_bet_fractions(game, info, pot_bb);
    let spr = if pot_bb > 0.0 {
        eff_stack_bb / pot_bb
    } else {
        eff_stack_bb
    };

    GameFeatures {
        n_players,
        position,
        street,
        eff_stack_bb,
        pot_bb,
        to_call_bb,
        bet_frac_pot,
        spr,
    }
}

/// Count active (non-folded) players
fn count_active_players(game: &Game) -> u8 {
    match game.turn() {
        Turn::Terminal => game
            .settlements()
            .iter()
            .filter(|s| s.status != State::Folding)
            .count() as u8,
        _ => game.n() as u8, // Approximate: assume no folds in non-terminal states
    }
}

/// Get the hero (current actor) index
fn get_hero_index(game: &Game) -> u8 {
    match game.turn() {
        Turn::Choice(p) => p as u8,
        _ => 0,
    }
}

/// Calculate position relative to button
fn calculate_position(game: &Game, hero_idx: u8) -> u8 {
    let dealer_pos = game.dealer();
    if hero_idx >= dealer_pos {
        hero_idx - dealer_pos
    } else {
        (game.n() as u8) + hero_idx - dealer_pos
    }
}

/// Encode street as numeric value
fn encode_street(street: Street) -> u8 {
    match street {
        Street::Pref => 0,
        Street::Flop => 1,
        Street::Turn => 2,
        Street::Rive => 3,
    }
}

/// Calculate bet sizes as fractions of pot
fn calculate_bet_fractions(game: &Game, info: &Info, pot_bb: f32) -> Vec<f32> {
    info.futures()
        .clone()
        .map(|edge| match edge {
            Edge::Raise(odds) => odds.0 as f32 / odds.1 as f32,
            Edge::Shove => {
                if pot_bb > 0.0 {
                    game.to_shove() as f32 / crate::B_BLIND as f32 / pot_bb
                } else {
                    100.0 // Large value for all-in when pot is empty
                }
            }
            _ => 0.0,
        })
        .collect()
}

/// Extract action sequence for the current street
fn extract_action_sequence(info: &Info, game: &Game) -> Vec<u8> {
    let current_street = game.street();
    let mut replay_game = Game::root();

    info.history()
        .clone()
        .filter_map(|edge| {
            let prev_street = replay_game.street();
            replay_game = replay_game.apply(replay_game.actionize(&edge));

            if prev_street == current_street && !edge.is_chance() {
                Some(u8::from(edge))
            } else {
                None
            }
        })
        .collect()
}

/// Generic helper to create list arrays
fn create_list_array<T, M>(vecs: &[Vec<T>]) -> Box<dyn Array>
where
    T: Copy,
    M: MutableArray + TryExtend<Option<T>> + Default,
{
    let mut list_array = MutableListArray::<i32, M>::new();
    for vec in vecs {
        let iter = vec.iter().copied().map(Some);
        list_array
            .try_extend(vec![Some(iter.collect::<Vec<_>>())])
            .expect("Failed to extend list array");
    }
    list_array.into_box()
}
