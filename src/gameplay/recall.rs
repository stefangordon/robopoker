use crate::cards::card::Card;
use crate::cards::deck::Deck;
use crate::cards::hand::Hand;
use crate::cards::hole::Hole;
use crate::cards::isomorphism::Isomorphism;
use crate::cards::observation::Observation;
use crate::gameplay::action::Action;
use crate::gameplay::edge::Edge;
use crate::gameplay::game::Game;
use crate::gameplay::odds::Odds;
use crate::gameplay::path::Path;
use crate::gameplay::turn::Turn;
use crate::mccfr::nlhe::encoder::BlueprintEncoder;
use crate::Chips;

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
        for action_from_path in self.path {
            // self.path here is Vec<Action> from Recall struct
            current_game_state = current_game_state.apply(action_from_path);
            game_states.push(current_game_state);
        }

        game_states.into_iter()
    }
}

impl Recall {
    /// Adjusts an action so that it is legal in the given game state.  In
    /// particular this fills in missing information for calls (amount 0) and
    /// draws ("DEAL" with no cards listed).
    ///
    /// When we see a draw with an empty hand we infer the exact cards that
    /// should be revealed from the `seen` observation, instead of taking a
    /// random legal draw (which would desynchronise the board and break
    /// blueprint look-ups).
    fn adjust_action(&self, game: &Game, action: Action) -> Action {
        match action {
            // Calls with amount 0 → fill in correct chips or downgrade to check.
            Action::Call(_) => {
                let call_amount = game.to_call();
                if call_amount == 0 {
                    // No chips required – treat as a check.
                    Action::Check
                } else {
                    // Even if the call commits the entire stack, leave it as Call so
                    // we distinguish an all-in *bet* (Shove) from an all-in *call*.
                    Action::Call(call_amount)
                }
            }
            // "DEAL" without explicit cards → derive the community cards the
            // hero has already seen.
            Action::Draw(hand) if hand.size() == 0 => {
                let street = game.street();
                // cards that *should* be revealed at this street, according to
                // the observation passed in the request.
                let reveal: Hand = self
                    .seen
                    .public()
                    .clone()
                    .skip(street.n_observed())
                    .take(street.n_revealed())
                    .collect::<Vec<Card>>()
                    .into();
                Action::Draw(reveal)
            }
            // For user-supplied raises/shoves that are outside the legal range
            // (e.g. "RAISE 1" when the min raise is 6) we mirror the logic in
            // `Game::act` so that the edge we encode matches the action that
            // will actually be applied to the game tree.
            Action::Raise(_) | Action::Shove(_) => {
                if game.is_allowed(&action) {
                    action
                } else {
                    game.find_nearest_action(&action).unwrap_or(action)
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
            _ => 0, // Default to position 0 if hero isn't a player position
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
        let mut filtered_deck = Deck::from(Hand::from(
            u64::from(Hand::from(deck)) & !(u64::from(observed_cards)),
        ));

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
            // Fill in any missing information before applying.
            let action_to_apply = self.adjust_action(&game, action);

            // Apply the adjusted action
            game = game.apply(action_to_apply);
        }

        game
    }

    pub fn path(&self) -> Path {
        let mut edges_vec = Vec::with_capacity(self.path.len());
        let mut game = self.root();

        let mut depth = 0usize; // raises since last Draw

        for &raw_action in self.path.iter() {
            let adj = self.adjust_action(&game, raw_action);

            let mut action_for_game = adj; // may be replaced for raises

            match adj {
                Action::Draw(_) => {
                    edges_vec.push(game.edgify(adj));
                    depth = 0; // new betting round
                    log::debug!(
                        "ENCODE draw   | street={:?} depth→0 edge={:?}",
                        game.street(),
                        edges_vec.last().unwrap()
                    );
                }
                Action::Raise(amount) => {
                    let grid = BlueprintEncoder::raises(&game, depth);
                    let odds = Odds::nearest_in(&grid, amount, game.pot());
                    edges_vec.push(Edge::Raise(odds));
                    depth += 1;

                    // Snap amount to exact grid bet so game state matches blueprint expectation
                    use crate::Probability;
                    let snapped_amount =
                        ((Probability::from(odds) * game.pot() as f32).round()) as Chips;
                    action_for_game = Action::Raise(snapped_amount.max(game.to_raise()));
                    log::debug!(
                        "ENCODE raise  | amount={} pot={} depth={} odds={:?} edge={:?}",
                        amount,
                        game.pot(),
                        depth - 1,
                        odds,
                        edges_vec.last().unwrap()
                    );
                }
                Action::Shove(_) => {
                    edges_vec.push(game.edgify(adj));
                    depth += 1;
                    log::debug!(
                        "ENCODE shove  | depth={} edge={:?}",
                        depth - 1,
                        edges_vec.last().unwrap()
                    );
                }
                _ => {
                    edges_vec.push(game.edgify(adj));
                    log::debug!(
                        "ENCODE other  | action={:?} depth={} edge={:?}",
                        adj,
                        depth,
                        edges_vec.last().unwrap()
                    );
                }
            }

            game = game.apply(action_for_game);
        }

        // Blueprint stores the most-recent edge in the least-significant nibble,
        // so we must reverse the chronological order before packing.
        edges_vec.into_iter().rev().collect::<Vec<_>>().into()
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

    /// Return the hero's seat index (0-based).  Defaults to 0 if `hero` is not
    /// a `Choice` variant – that situation should not occur in normal use.
    pub fn hero_position(&self) -> usize {
        match self.hero {
            Turn::Choice(idx) => idx,
            _ => 0,
        }
    }
}
