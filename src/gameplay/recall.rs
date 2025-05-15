use crate::cards::card::Card;
use crate::cards::hole::Hole;
use crate::cards::isomorphism::Isomorphism;
use crate::cards::observation::Observation;
use crate::gameplay::action::Action;
use crate::gameplay::game::Game;
use crate::gameplay::path::Path;
use crate::gameplay::turn::Turn;
use crate::cards::hand::Hand;
use crate::cards::deck::Deck;

/// a complete representation of perfect recall game history
/// from the perspective of the hero. intended use is for
/// the path to be populated only with choice actions,
/// since we internally keep track of chance actions
/// by conditioning on the observed cards.
///
/// note that this struct implicitly assumes:
/// - default stacks
/// - default blinds
/// - default dealer position
/// - unordered private and community cards
#[derive(Debug, Clone)]
pub struct Recall {
    hero: Turn,
    seen: Observation, // could be replaced by Hole + Board + BetHistory(Vec<Action>)
    path: Vec<Action>,
}

impl From<(Turn, Observation, Vec<Action>)> for Recall {
    fn from((hero, seen, path): (Turn, Observation, Vec<Action>)) -> Self {
        Self { hero, seen, path }
    }
}

impl IntoIterator for Recall {
    type Item = Game;
    type IntoIter = std::vec::IntoIter<Game>;
    // Option 1: Using the clearer loop (recommended for review)
    fn into_iter(self) -> Self::IntoIter {
        let root_game = self.root();

        if self.path.is_empty() {
            return vec![root_game].into_iter();
        }

        let mut game_states: Vec<Game> = Vec::with_capacity(self.path.len() + 1);
        game_states.push(root_game);

        let mut current_game_state = root_game;
        for action_from_path in self.path { // self.path here is Vec<Action> from Recall struct
            current_game_state = current_game_state.apply(action_from_path);
            game_states.push(current_game_state);
        }

        game_states.into_iter()
    }
}

impl Recall {
    /// Adjusts an action so that it is legal in the given game state, performing
    /// the same auto-fill logic as used by `head()`.
    fn adjust_action(game: &Game, action: Action) -> Action {
        match action {
            Action::Call(_) => {
                let call_amount = game.to_call();
                if call_amount > 0 {
                    Action::Call(call_amount)
                } else {
                    Action::Check
                }
            }
            Action::Draw(hand) if hand.size() == 0 => {
                if let Some(Action::Draw(cards)) = game
                    .legal()
                    .into_iter()
                    .find(|a| matches!(a, Action::Draw(_)))
                {
                    Action::Draw(cards)
                } else {
                    Action::Draw(hand) // keep original
                }
            }
            other => other,
        }
    }

    pub fn new(seen: Observation, hero: Turn) -> Self {
        Self {
            seen,
            hero,
            path: Game::blinds(),
        }
    }

    pub fn root(&self) -> Game {
        // Get the hero's position
        let hero_idx = match self.hero {
            Turn::Choice(idx) => idx,
            _ => 0  // Default to position 0 if hero isn't a player position
        };

        // Start with a basic game
        let mut game = Game::root();

        // 1. Set hero's cards from observation
        game = game.wipe(Hole::from(self.seen), hero_idx);

        // 2. Create a set of safe placeholder cards for opponents
        // First, identify all observed cards (both hero's hole cards and community cards)
        let observed_cards = Hand::from(self.seen);

        // Get a fresh deck and remove all observed cards
        let deck = Deck::new();
        // Create a new deck that doesn't contain the observed cards
        let mut filtered_deck = Deck::from(Hand::from(u64::from(Hand::from(deck)) & !(u64::from(observed_cards))));

        // 3. For opponent seats, reset their cards to safe placeholder cards
        for i in 0..game.n() {
            if i != hero_idx {
                // Skip the hero's seat, handle all opponents
                // Create a safe hole for this opponent from remaining cards in the deck
                let safe_hole = filtered_deck.hole();
                game = game.reset_cards_at(i, safe_hole);
            }
        }

        game
    }

    pub fn head(&self) -> Game {
        let mut game = self.root();

        for action in self.path.iter().cloned() {
            // Special handling for calls with amount 0
            let action_to_apply = Self::adjust_action(&game, action);

            // Apply the adjusted action
            game = game.apply(action_to_apply);
        }

        game
    }

    pub fn path(&self) -> Path {
        let mut edges_vec = Vec::with_capacity(self.path.len());
        let mut game = self.root();

        for &raw_action in self.path.iter() {
            let adj = Self::adjust_action(&game, raw_action);
            edges_vec.push(game.edgify(adj));
            game = game.apply(adj);
        }

        edges_vec.into()
    }

    pub fn isomorphism(&self) -> Isomorphism {
        Isomorphism::from(self.seen)
    }

    pub fn undo(&mut self) {
        if self.can_rewind() {
            self.path.pop();
        }
        while self.can_revoke() {
            self.path.pop();
        }
    }
    pub fn push(&mut self, action: Action) {
        if self.can_extend(&action) {
            self.path.push(action);
        }
        while self.can_reveal() {
            let street = self.head().street();
            let reveal = self
                .seen
                .public()
                .clone()
                .skip(street.n_observed())
                .take(street.n_revealed())
                .collect::<Vec<Card>>()
                .into();
            self.path.push(Action::Draw(reveal));
        }
    }

    pub fn can_extend(&self, action: &Action) -> bool {
        self.head().is_allowed(action)
    }
    pub fn can_rewind(&self) -> bool {
        self.path.iter().any(|a| !a.is_blind())
    }
    pub fn can_lookup(&self) -> bool {
        true
            && self.head().turn() == self.hero //               is it our turn right now?
            && self.head().street() == self.seen.street() //    have we exhausted info from Obs?
    }
    pub fn can_reveal(&self) -> bool {
        true
            && self.head().turn() == Turn::Chance //            is it time to reveal the next card?
            && self.head().street() < self.seen.street() //     would revealing double-deal?
    }
    pub fn can_revoke(&self) -> bool {
        matches!(self.path.last().expect("empty path"), Action::Draw(_))
    }
}
