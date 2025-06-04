use super::edge::Edge;
use super::game::Game;
use super::solver::RPS;
use super::turn::Turn;
use crate::gameplay::game::Game as GameplayGame;
use crate::gameplay::odds::Odds;
use crate::mccfr::nlhe::encoder::BetSizer;
use crate::mccfr::structs::tree::Tree;
use crate::mccfr::types::branch::Branch;

/// RPS doesn't need bet sizing since it's a simple game with fixed actions
pub struct RPSSizer;
impl BetSizer for RPSSizer {
    fn raises(_game: &GameplayGame, _depth: usize) -> Vec<Odds> {
        vec![] // RPS doesn't have any raises
    }
}

impl crate::mccfr::traits::encoder::Encoder for RPS {
    type T = Turn;
    type E = Edge;
    type G = Game;
    type I = Turn;
    type S = RPSSizer;

    fn seed(&self, _: &Self::G) -> Self::I {
        Turn::P1
    }

    fn info(
        &self,
        _: &Tree<Self::T, Self::E, Self::G, Self::I>,
        (_, game, _): Branch<Self::E, Self::G>,
    ) -> Self::I {
        use crate::mccfr::traits::game::Game;
        game.turn()
    }
}
