use super::{Edge, Game, Info, Turn, SUBGAME_RAISES};
use crate::cards::isomorphism::Isomorphism;
use crate::clustering::abstraction::Abstraction;
use crate::gameplay::action::Action;
use crate::gameplay::path::Path;
use crate::mccfr::structs::tree::Tree;
use crate::mccfr::traits::encoder::Encoder;
use crate::mccfr::types::branch::Branch;
use std::collections::BTreeMap;

/// Encoder for subgame solving with enhanced action abstraction
pub struct SubgameEncoder {
    /// Abstraction lookup table (same as blueprint)
    lookup: BTreeMap<Isomorphism, Abstraction>,
}

impl SubgameEncoder {
    pub fn new(lookup: BTreeMap<Isomorphism, Abstraction>) -> Self {
        Self { lookup }
    }

    /// Get the available raise sizes for the current game state
    fn subgame_raises(depth: usize) -> Vec<crate::gameplay::odds::Odds> {
        if depth > crate::MAX_RAISE_REPEATS {
            vec![]
        } else {
            SUBGAME_RAISES.to_vec()
        }
    }

    /// Convert legal actions to edges with enhanced abstraction
    fn subgame_choices(game: &Game, depth: usize) -> Vec<Edge> {
        game.legal()
            .into_iter()
            .flat_map(|action| match action {
                Action::Raise(_) => Self::subgame_raises(depth)
                    .into_iter()
                    .map(Edge::from)
                    .collect(),
                _ => vec![Edge::from(action)],
            })
            .collect()
    }

    fn abstraction(&self, iso: &Isomorphism) -> Abstraction {
        self.lookup
            .get(iso)
            .copied()
            .unwrap_or_else(|| {
                // Fallback: use the raw isomorphism value as abstraction
                // This works for river where abstraction == isomorphism
                Abstraction::from(i64::from(*iso))
            })
    }
}

impl Encoder for SubgameEncoder {
    type T = Turn;
    type E = Edge;
    type G = Game;
    type I = Info;

    fn seed(&self, root: &Self::G) -> Self::I {
        let iso = Isomorphism::from(root.sweat());
        let present = self.abstraction(&iso);
        let futures = Path::from(Self::subgame_choices(root, 0));
        Self::I::from((Path::default(), present, futures))
    }

    fn info(&self, tree: &Tree<Self::T, Self::E, Self::G, Self::I>, leaf: Branch<Self::E, Self::G>) -> Self::I {
        let (edge, ref game, head) = leaf;
        let iso = Isomorphism::from(game.sweat());
        
        // Count raises in current betting round to determine depth
        let n_raises = tree
            .at(head)
            .into_iter()
            .take_while(|(_, e)| e.is_choice())
            .filter(|(_, e)| e.is_aggro())
            .count();
            
        let present = self.abstraction(&iso);
        let futures = Path::from(Self::subgame_choices(game, n_raises));
        
        // Collect history path (limited to MAX_DEPTH_SUBGAME)
        let history = std::iter::once(edge)
            .chain(tree.at(head).into_iter().map(|(_, e)| e))
            .take(crate::MAX_DEPTH_SUBGAME)
            .collect::<Path>();
            
        Self::I::from((history, present, futures))
    }
} 