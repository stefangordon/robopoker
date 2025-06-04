use super::evaluator::Evaluator;
use super::hand::Hand;
use super::kicks::Kickers;
use super::rank::Rank;
use super::ranking::Ranking;
use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::path::PathBuf;
use std::sync::OnceLock;

/// A faster 7-card hand evaluator backed by a pre-computed lookup table.
///
/// The table maps every 7-card combination (52 choose 7 ≈ 134 M) to a compact
/// 16-bit encoding of the hand `Ranking`.  On first use we attempt to load the
/// table from `pgcopy/eval7.bin`.  If the file is missing we generate it from
/// the existing bit-twiddling [`Evaluator`].  This generation step is *slow* –
/// it enumerates all 7-card combinations once – but it only happens when the
/// file is absent (typically during the first clustering run).  Subsequent
/// usages memory-map / read the table and perform O(1) in-memory look-ups.
pub struct LookupEvaluator(Hand);

impl From<Hand> for LookupEvaluator {
    fn from(h: Hand) -> Self {
        Self(h)
    }
}

impl LookupEvaluator {
    /* ---------- public API compatible with Evaluator ---------- */

    #[inline]
    pub fn find_ranking(&self) -> Ranking {
        // Fallback to the original slow path if the hand has <7 cards.  This
        // keeps behaviour identical to the existing Evaluator in those cases.
        if self.0.size() < 7 {
            return Evaluator::from(self.0).find_ranking();
        }

        // Fast path – 7-card lookup.
        let idx = combination_index(u64::from(self.0));
        let table = Self::table();
        let enc = table[idx];
        decode_ranking(enc)
    }

    #[inline]
    pub fn find_kickers(&self, ranking: Ranking) -> Kickers {
        // Kickers are inexpensive compared to the ranking itself.  Re-use the
        // existing logic for full correctness.
        Evaluator::from(self.0).find_kickers(ranking)
    }

    /* ---------- table initialisation ---------- */

    fn table() -> &'static [u16] {
        static CELL: OnceLock<Vec<u16>> = OnceLock::new();
        CELL.get_or_init(|| load_or_generate_table())
    }
}

/* =============================================================================
                                Encoding helpers
=============================================================================*/

fn encode_ranking(r: Ranking) -> u16 {
    // Layout: 0..3  = hi rank index (4 bits)
    //         4..7  = lo rank index (4 bits)
    //         8..11 = variant (4 bits)
    let (variant, hi, lo): (u16, u16, u16) = match r {
        Ranking::HighCard(hi) => (0, u8::from(hi) as u16, 0),
        Ranking::OnePair(hi) => (1, u8::from(hi) as u16, 0),
        Ranking::TwoPair(hi, lo) => (2, u8::from(hi) as u16, u8::from(lo) as u16),
        Ranking::ThreeOAK(hi) => (3, u8::from(hi) as u16, 0),
        Ranking::Straight(hi) => (4, u8::from(hi) as u16, 0),
        Ranking::FullHouse(hi, lo) => (5, u8::from(hi) as u16, u8::from(lo) as u16),
        Ranking::Flush(hi) => (6, u8::from(hi) as u16, 0),
        Ranking::FourOAK(hi) => (7, u8::from(hi) as u16, 0),
        Ranking::StraightFlush(hi) => (8, u8::from(hi) as u16, 0),
        Ranking::MAX => (15, 0, 0),
    };
    (variant << 8) | ((lo & 0xF) << 4) | (hi & 0xF)
}

#[inline(always)]
fn decode_ranking(bits: u16) -> Ranking {
    let variant = (bits >> 8) & 0xF;
    let hi = Rank::from((bits & 0xF) as u8);
    let lo = Rank::from(((bits >> 4) & 0xF) as u8);
    match variant {
        0 => Ranking::HighCard(hi),
        1 => Ranking::OnePair(hi),
        2 => Ranking::TwoPair(hi, lo),
        3 => Ranking::ThreeOAK(hi),
        4 => Ranking::Straight(hi),
        5 => Ranking::FullHouse(hi, lo),
        6 => Ranking::Flush(hi),
        7 => Ranking::FourOAK(hi),
        8 => Ranking::StraightFlush(hi),
        15 => Ranking::MAX,
        _ => unreachable!("invalid ranking encoding: {bits}"),
    }
}

/* =============================================================================
                       Combination indexing (52 choose 7)
=============================================================================*/

const HAND_SIZE: usize = 7;
const TOTAL_COMBOS: usize = 133_784_560; // 52 choose 7

fn combination_index(bits: u64) -> usize {
    debug_assert_eq!(bits.count_ones(), HAND_SIZE as u32);

    let mut idx = 0usize;
    let mut r = 1usize; // r ranges 1..=HAND_SIZE as per combinatorial number system
    for card in 0..52 {
        if bits & (1u64 << card) != 0 {
            idx += n_choose_k(card, r);
            r += 1;
            if r > HAND_SIZE {
                break;
            }
        }
    }
    idx
}

fn n_choose_k(n: usize, k: usize) -> usize {
    if n < k {
        return 0;
    }
    // Simple, small k (<=7) – no risk of overflow with 64-bit intermediates.
    let mut result = 1usize;
    let mut kk = 1usize;
    while kk <= k {
        result = result * (n + 1 - kk) / kk;
        kk += 1;
    }
    result
}

/* =============================================================================
                       Table generation / persistence helpers
=============================================================================*/

fn load_or_generate_table() -> Vec<u16> {
    let path = table_path();
    if path.exists() {
        log::info!("loading 7-card evaluator table from {}", path.display());
        load_table(&path)
    } else {
        log::warn!(
            "evaluator table not found – generating at {} (this may take a while)",
            path.display()
        );
        let table = generate_table();
        save_table(&path, &table);
        table
    }
}

fn table_path() -> PathBuf {
    let cwd = std::env::current_dir().unwrap_or_default();
    cwd.join("pgcopy").join("eval7.bin")
}

fn load_table(path: &PathBuf) -> Vec<u16> {
    let mut reader = BufReader::new(File::open(path).expect("open eval7.bin"));
    let mut buf = Vec::new();
    reader.read_to_end(&mut buf).expect("read eval7.bin");
    assert_eq!(buf.len() % 2, 0, "corrupt eval7.bin file");
    buf.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

fn save_table(path: &PathBuf, table: &[u16]) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let mut file = File::create(path).expect("create eval7.bin");
    for &val in table {
        file.write_all(&val.to_le_bytes()).expect("write eval7.bin");
    }
    file.flush().ok();
}

fn generate_table() -> Vec<u16> {
    use crate::cards::evaluator::Evaluator;
    use crate::cards::hands::HandIterator;

    let mut table = vec![0u16; TOTAL_COMBOS];
    let mut processed: usize = 0;

    for hand in HandIterator::from((HAND_SIZE, Hand::empty())) {
        let bits = u64::from(hand);
        let idx = combination_index(bits);
        let ranking = Evaluator::from(hand).find_ranking();
        table[idx] = encode_ranking(ranking);

        processed += 1;
        if processed % 5_000_000 == 0 {
            log::info!("generated {}/{} entries", processed, TOTAL_COMBOS);
        }
    }
    table
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cards::{evaluator::Evaluator, hands::HandIterator};

    // Verify that the lookup evaluator produces identical rankings to the
    // original evaluator for a sample of 10,000 distinct 7-card hands.
    #[test]
    fn lookup_matches_original_on_seven_cards() {
        const SAMPLE: usize = 10_000;
        let mut iter = HandIterator::from((7, Hand::empty()));
        for (i, hand) in iter.by_ref().take(SAMPLE).enumerate() {
            let lookup_rank = LookupEvaluator::from(hand).find_ranking();
            let original_rank = Evaluator::from(hand).find_ranking();
            assert_eq!(
                lookup_rank, original_rank,
                "Mismatch at sample {} for hand {}",
                i, hand
            );
        }
    }

    // Spot-check that hands with <7 cards fall back to the classic evaluator.
    #[test]
    fn fallback_for_smaller_hands() {
        let hand_5 = Hand::try_from("As Ks Qd Jc 9s").unwrap();
        let lookup_rank = LookupEvaluator::from(hand_5).find_ranking();
        let original_rank = Evaluator::from(hand_5).find_ranking();
        assert_eq!(lookup_rank, original_rank);
    }
}
