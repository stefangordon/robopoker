#![allow(dead_code)]

use super::action::Action;
use super::seat::Seat;
use super::seat::State;
use super::settlement::Settlement;
use super::showdown::Showdown;
use super::turn::Turn;
use crate::cards::board::Board;
use crate::cards::deck::Deck;
use crate::cards::hand::Hand;
use crate::cards::hole::Hole;
use crate::cards::observation::Observation;
use crate::cards::street::Street;
use crate::cards::strength::Strength;
use crate::Chips;
use crate::N;
use crate::STACK;

type Position = u8;
/// Rotation represents the memoryless state of the game in between actions.
///
/// It records both public and private data structs, and is responsible for managing the
/// rotation of players, the pot, and the board. Its immutable methods reveal
/// pure functions representing the rules of how the game may proceed.
/// This full game state will also be our CFR node representation.
#[derive(Debug, Clone, Copy)]
pub struct Game {
    seats: [Seat; N],
    board: Board,
    pot: Chips,
    dealer: Position,
    ticker: Position,
}

/// we enable different start points of the game tree,
/// depending on whether or not blinds should be posted
/// or cards should be dealt.
impl Game {
    /// this will start the game at the first decision
    /// NOT the first action, which are blinds and hole cards dealt.
    /// stack size is always 100 and P1 is always dealer.
    /// these should not matter too much in the MCCFR algorithm,
    /// as long as we alternate the traverser/paths explored
    pub fn root() -> Self {
        Self::base().deal().post()
    }
    pub fn base() -> Self {
        Self {
            seats: [Seat::from(STACK); N],
            board: Board::empty(),
            pot: Chips::from(0i16),
            dealer: 0u8,
            ticker: if N == 2 { 0u8 } else { 1u8 },
        }
    }
    pub fn deal(mut self) -> Self {
        self.deal_cards();
        self
    }
    pub fn post(mut self) -> Self {
        self.act(Action::Blind(self.to_post()));
        self.act(Action::Blind(self.to_post()));
        self
    }
    pub fn wipe(mut self, hole: Hole, hero_idx: usize) -> Self {
        if hero_idx < self.seats.len() {
            self.seats[hero_idx].reset_cards(hole);
        }
        self
    }
}

/// read-only state variabes are exposed publicly
/// so that the plethora of bool fn's that determine
/// game state are obfuscated from the caller
impl Game {
    pub fn n(&self) -> usize {
        self.seats.len()
    }
    pub fn pot(&self) -> Chips {
        self.pot
    }
    pub fn board(&self) -> Board {
        self.board
    }
    /// Get the dealer position (button) for the current game
    pub fn dealer(&self) -> Position {
        self.dealer
    }
    pub fn turn(&self) -> Turn {
        if self.must_stop() {
            Turn::Terminal
        } else if self.must_deal() {
            Turn::Chance
        } else {
            Turn::Choice(self.actor_idx())
        }
    }
    pub fn actor(&self) -> &Seat {
        self.actor_ref()
    }
    pub fn sweat(&self) -> Observation {
        Observation::from((
            Hand::from(self.actor().cards()), //
            Hand::from(self.board()),         //
        ))
    }
    pub fn street(&self) -> Street {
        self.board.street()
    }
}

/// the game rules are encoded in the set of
/// legal moves available and a bunch of
/// bool fns that determine action validity from
/// immutable reference
impl Game {
    pub fn apply(&self, action: Action) -> Self {
        let mut child = self.clone();

        // Apply the action directly without special adjustment
        child.act(action);
        child
    }
    pub fn legal(&self) -> Vec<Action> {
        let mut options = Vec::new();
        if self.must_stop() {
            return options;
        }
        if self.must_deal() {
            options.push(Action::Draw(self.deck().deal(self.street())));
            return options;
        }
        if self.must_post() {
            options.push(Action::Blind(Self::sblind()));
            return options;
        }
        if self.may_raise() {
            options.push(Action::Raise(self.to_raise()));
        }
        if self.may_shove() {
            options.push(Action::Shove(self.to_shove()));
        }
        if self.may_call() {
            options.push(Action::Call(self.to_call()));
        }
        if self.may_fold() {
            options.push(Action::Fold);
        }
        if self.may_check() {
            options.push(Action::Check);
        }
        assert!(options.len() > 0);
        options
    }
    pub fn is_allowed(&self, action: &Action) -> bool {
        if self.must_stop() {
            return false;
        }
        match action {
            Action::Raise(raise) => {
                self.may_raise()
                    && raise.clone() >= self.to_raise()
                    && raise.clone() <= self.to_shove() - 1
            }
            Action::Draw(cards) => {
                self.must_deal()
                    && cards.clone().all(|c| self.deck().contains(&c))
                    && cards.count() == self.board().street().n_revealed()
            }
            Action::Blind(_) => self.must_post(),
            _ => self.legal().contains(action),
        }
    }
}

/// mutating methods modify game state privately
/// such that we enforce NLHE invariants irrespective
/// of caller behavior
impl Game {
    fn conclude(&mut self) {
        self.give_chips();
    }
    fn commence(&mut self) {
        assert!(self.seats.iter().all(|s| s.stack() > 0), "game over");
        self.wipe_board();
        self.deal_cards();
        self.move_button();
        self.act(Action::Blind(self.to_post()));
        self.act(Action::Blind(self.to_post()));
    }
    fn give_chips(&mut self) {
        for (_, (settlement, seat)) in self
            .settlements()
            .iter()
            .zip(self.seats.iter_mut())
            .enumerate()
            .inspect(|(i, (x, s))| log::trace!("{} {} {:>7} {}", i, s.cards(), s.stack(), x.pnl()))
        {
            seat.win(settlement.reward);
        }
    }
    fn wipe_board(&mut self) {
        self.pot = 0;
        self.board.clear();
        assert!(self.street() == Street::Pref);
    }
    fn move_button(&mut self) {
        assert!(self.seats.len() == self.n());
        assert!(self.street() == Street::Pref);
        self.dealer = ((self.dealer as usize + 1) % self.n()) as Position;
        self.ticker = self.dealer;
        self.next_player();
    }
    fn deal_cards(&mut self) {
        assert!(self.street() == Street::Pref);
        let mut deck = Deck::new();
        for seat in self.seats.iter_mut() {
            seat.reset_state(State::Betting);
            seat.reset_cards(deck.hole());
            seat.reset_stake();
            seat.reset_spent();
        }
    }
    fn act(&mut self, a: Action) {
        // Handle special cases already in our action
        let adjusted_action = match a {
            Action::Raise(_) | Action::Shove(_) => {
                if !self.is_allowed(&a) {
                    if let Some(nearest) = self.find_nearest_action(&a) {
                        log::info!("Adjusting {:?} to nearest allowed value: {:?}", a, nearest);
                        nearest
                    } else {
                        a
                    }
                } else {
                    a
                }
            }
            Action::Draw(hand) if hand.size() == 0 => {
                // Handle "DEAL" without explicit cards - deal the appropriate cards for this street
                if self.must_deal() {
                    Action::Draw(self.deck().deal(self.street()))
                } else {
                    a
                }
            }
            _ => a,
        };

        // Check if the adjusted action is allowed
        if !self.is_allowed(&adjusted_action) {
            // If still not allowed, error
            let error_msg = self.debug_action_error(&adjusted_action);
            log::error!("{}", error_msg);
            panic!("{}", error_msg);
        }

        match adjusted_action {
            Action::Check => {
                self.next_player();
            }
            Action::Fold => {
                self.fold();
                self.next_player();
            }
            Action::Call(chips)
            | Action::Blind(chips)
            | Action::Raise(chips)
            | Action::Shove(chips) => {
                self.bet(chips);
                self.next_player();
            }
            Action::Draw(cards) => {
                self.show(cards);
                self.next_player();
                self.next_street();
            }
        }
    }
    fn bet(&mut self, bet: Chips) {
        assert!(self.actor_ref().stack() >= bet);
        self.pot += bet;
        self.actor_mut().bet(bet);
        if self.actor_ref().stack() == 0 {
            self.shove();
        }
    }
    fn shove(&mut self) {
        self.actor_mut().reset_state(State::Shoving);
    }
    fn fold(&mut self) {
        self.actor_mut().reset_state(State::Folding);
    }
    fn show(&mut self, hand: Hand) {
        self.ticker = self.dealer;
        self.board.add(hand);
    }
}

/// advancing to the next street or player
/// is privatized, since the only public
/// facing interface is applying actions
/// to transition the state machine
impl Game {
    fn next_street(&mut self) {
        for seat in self.seats.iter_mut() {
            seat.reset_stake();
        }
    }
    fn next_player(&mut self) {
        if !self.is_everyone_alright() {
            loop {
                self.ticker = self.ticker.wrapping_add(1);
                match self.actor_ref().state() {
                    State::Betting => break,
                    State::Folding => continue,
                    State::Shoving => continue,
                }
            }
        }
    }
}

/// boolean constraints that are composed
/// from one another
impl Game {
    /// we're waiting for showdown or everyone folded
    fn must_stop(&self) -> bool {
        // Count seats that have not folded
        let active = self
            .seats
            .iter()
            .filter(|s| s.state() != State::Folding)
            .count();

        // If every seat has folded (active == 0) we have entered an
        // impossible terminal state where no one can win the pot.  Do
        // NOT treat this as hand-finished; let the state machine advance
        // until at least one seat remains so chips are awarded correctly.
        if active == 0 {
            return false;
        }

        if self.street() == Street::Rive {
            self.is_everyone_alright()
        } else {
            // Pre-river we stop as soon as exactly one seat remains.
            self.is_everyone_folding()
        }
    }
    /// we're waiting for a card to be revealed
    fn must_deal(&self) -> bool {
        if self.street() == Street::Rive {
            false
        } else {
            self.is_everyone_alright()
        }
    }
    /// blinds have not yet been posted // TODO some edge case of all in blinds
    fn must_post(&self) -> bool {
        if self.street() == Street::Pref {
            self.pot() < Self::sblind() + Self::bblind()
        } else {
            false
        }
    }

    /// all players have acted, the pot is right.
    fn is_everyone_alright(&self) -> bool {
        self.is_everyone_calling() || self.is_everyone_folding() || self.is_everyone_shoving()
    }
    /// all players betting are in for the same amount
    fn is_everyone_calling(&self) -> bool {
        self.is_everyone_touched() && self.is_everyone_matched()
    }
    /// all players have acted at least once
    fn is_everyone_touched(&self) -> bool {
        let pref_offset = if self.street() == Street::Pref {
            if self.n() == 2 {
                1
            } else {
                2
            }
        } else {
            0
        };

        // Use `>` so each active player still gets their turn before the
        // state machine advances to the next street.
        (self.ticker as usize) > self.n() + pref_offset
    }
    /// all players betting are in for the effective stake
    fn is_everyone_matched(&self) -> bool {
        let stake = self.effective_stake();
        self.seats
            .iter()
            .filter(|s| s.state() == State::Betting)
            .all(|s| s.stake() == stake)
    }
    /// all players betting or shoving are shoving
    fn is_everyone_shoving(&self) -> bool {
        self.seats
            .iter()
            .filter(|s| s.state() != State::Folding)
            .all(|s| s.state() == State::Shoving)
    }
    /// there is exactly one player betting or shoving
    fn is_everyone_folding(&self) -> bool {
        self.seats
            .iter()
            .filter(|s| s.state() != State::Folding)
            .count()
            == 1
    }

    //
    fn may_fold(&self) -> bool {
        self.to_call() > 0
    }
    fn may_call(&self) -> bool {
        self.may_fold() && self.to_call() < self.to_shove()
    }
    fn may_check(&self) -> bool {
        self.effective_stake() == self.actor_ref().stake()
    }
    fn may_raise(&self) -> bool {
        self.to_raise() < self.to_shove()
    }
    fn may_shove(&self) -> bool {
        self.to_shove() > 0
    }
}

/// the chip constraints imposed at a given game state
/// are calculated and exposed for UI purposes
impl Game {
    pub fn to_call(&self) -> Chips {
        self.effective_stake() - self.actor_ref().stake()
    }
    pub fn to_post(&self) -> Chips {
        assert!(self.street() == Street::Pref);

        let offset = (self.ticker as isize - self.dealer as isize).rem_euclid(self.n() as isize);

        // In heads-up the dealer (offset 0) posts SB and the other player (offset 1) posts BB.
        // In 3-way+ games the seat one step left of the button (offset 1) posts SB and
        // the seat two steps left (offset 2) posts BB.
        if self.n() == 2 {
            match offset {
                0 => Self::sblind().min(self.actor_ref().stack()), // Dealer – SB
                _ => Self::bblind().min(self.actor_ref().stack()), // Opponent – BB
            }
        } else {
            match offset {
                1 => Self::sblind().min(self.actor_ref().stack()), // SB
                _ => Self::bblind().min(self.actor_ref().stack()), // BB or others (straddle not supported)
            }
        }
    }
    pub fn to_shove(&self) -> Chips {
        self.actor_ref().stack()
    }
    pub fn to_raise(&self) -> Chips {
        let (most_large_stake, next_large_stake) = self
            .seats
            .iter()
            .filter(|s| s.state() != State::Folding)
            .map(|s| s.stake())
            .fold((0, 0), |(most, next), stake| {
                if stake > most {
                    (stake, most)
                } else if stake > next {
                    (most, stake)
                } else {
                    (most, next)
                }
            });
        let relative_raise = most_large_stake - self.actor().stake();
        let marginal_raise = most_large_stake - next_large_stake;
        let required_raise = std::cmp::max(marginal_raise, Self::bblind());
        relative_raise + required_raise
    }
}

/// payout calculations
impl Game {
    pub fn settlements(&self) -> Vec<Settlement> {
        assert!(self.must_stop(), "non terminal game state:\n{}", self);
        Showdown::from(self.ledger()).settle()
    }
    fn ledger(&self) -> Vec<Settlement> {
        self.seats
            .iter()
            .map(|seat| self.entry(seat))
            .collect::<Vec<Settlement>>()
    }
    fn entry(&self, seat: &Seat) -> Settlement {
        Settlement {
            reward: 0,
            risked: seat.spent(),
            status: seat.state(),
            strength: self.strength(seat),
        }
    }
    fn strength(&self, seat: &Seat) -> Strength {
        Strength::from(Hand::add(
            Hand::from(seat.cards()),
            Hand::from(self.board()),
        ))
    }
}

/// card calculations
impl Game {
    pub fn draw(&self) -> Hand {
        self.deck().deal(self.street())
    }
    pub fn deck(&self) -> Deck {
        let mut removed = Hand::from(self.board);
        for seat in self.seats.iter() {
            let hole = Hand::from(seat.cards());
            removed = Hand::add(removed, hole);
        }
        Deck::from(removed.complement())
    }
}

/// current position / actor calculations
impl Game {
    fn actor_idx(&self) -> usize {
        ((self.dealer as usize) + (self.ticker as usize)) % self.n()
    }
    fn actor_ref(&self) -> &Seat {
        let index = self.actor_idx();
        self.seats
            .get(index)
            .expect("index should be in bounds bc modulo")
    }
    fn actor_mut(&mut self) -> &mut Seat {
        let index = self.actor_idx();
        self.seats
            .get_mut(index)
            .expect("index should be in bounds bc modulo")
    }
}

/// effective bet sizes
impl Game {
    /// Calculate the effective stack size (second-largest stack among all players)
    /// This represents the maximum amount that can be won/lost in the hand
    pub fn effective_stack(&self) -> Chips {
        let mut totals = self
            .seats
            .iter()
            .map(|s| s.stack() + s.stake())
            .collect::<Vec<Chips>>();
        totals.sort_unstable();
        totals.pop().unwrap_or(0);
        totals.pop().unwrap_or(0)
    }
    fn effective_stake(&self) -> Chips {
        self.seats
            .iter()
            .map(|s| s.stake())
            .max()
            .expect("non-empty seats")
    }

    /// Total chips the given player has already committed to the pot (across all streets).
    /// This is a lightweight accessor used mainly by evaluation code; it never mutates state.
    pub fn spent(&self, position: usize) -> Chips {
        self.seats
            .get(position)
            .map(|seat| seat.spent())
            .expect("player index in bounds")
    }
}

/// define blinds
impl Game {
    pub fn blinds() -> Vec<Action> {
        vec![Action::Blind(Self::sblind()), Action::Blind(Self::bblind())]
    }
    pub const fn bblind() -> Chips {
        crate::B_BLIND
    }
    pub const fn sblind() -> Chips {
        crate::S_BLIND
    }
}

// odds and tree building stuff
impl Game {
    /// convert an Edge into an Action by using Game state to
    /// determine free parameters (stack size, pot size, etc)
    ///
    /// NOTE
    /// this conversion is not injective, as multiple edges may
    /// represent the same action. moreover, we "snap" raises to be
    /// within range of legal bet sizes, so sometimes Raise(5:1) yields
    /// an identical Game node as Raise(1:1) or Shove.
    pub fn actionize(&self, edge: &crate::gameplay::edge::Edge) -> Action {
        let game = self;
        match &edge {
            crate::gameplay::edge::Edge::Check => Action::Check,
            crate::gameplay::edge::Edge::Fold => Action::Fold,
            crate::gameplay::edge::Edge::Draw => Action::Draw(game.draw()),
            crate::gameplay::edge::Edge::Call => Action::Call(game.to_call()),
            crate::gameplay::edge::Edge::Shove => Action::Shove(game.to_shove()),
            crate::gameplay::edge::Edge::Raise(odds) => {
                let min = game.to_raise();
                let max = game.to_shove();
                let pot = game.pot() as crate::Utility;
                let odd = crate::Utility::from(*odds);
                let bet = (pot * odd) as Chips;
                match bet {
                    bet if bet >= max => Action::Shove(max),
                    bet if bet <= min => Action::Raise(min),
                    _ => Action::Raise(bet),
                }
            }
        }
    }

    pub fn edgify(&self, action: Action) -> crate::gameplay::edge::Edge {
        use crate::gameplay::edge::Edge;
        use crate::gameplay::odds::Odds;
        match action {
            Action::Fold => Edge::Fold,
            Action::Check => Edge::Check,
            Action::Draw(_) => Edge::Draw,
            Action::Shove(_) => Edge::Shove,
            Action::Blind(_) | Action::Call(_) => Edge::Call,
            Action::Raise(amount) => Edge::Raise(Odds::nearest((amount, self.pot()))),
        }
    }
}

impl crate::mccfr::traits::game::Game for Game {
    type E = crate::gameplay::edge::Edge;
    type T = crate::gameplay::turn::Turn;
    fn root() -> Self {
        Self::root()
    }
    fn turn(&self) -> Self::T {
        self.turn()
    }
    fn apply(&self, edge: Self::E) -> Self {
        self.apply(self.actionize(&edge))
    }
    fn payoff(&self, turn: Self::T) -> crate::Utility {
        self.settlements()
            .get(turn.position())
            .map(|settlement| settlement.pnl() as crate::Utility)
            .expect("player index in bounds")
    }
}

impl std::fmt::Display for Game {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let string = format!(" @ {:>6} {} {}", self.pot, self.board, self.street());
        for seat in self.seats.iter() {
            write!(f, "{}{:<6}", seat.state(), seat.stack())?;
        }
        #[cfg(feature = "native")]
        {
            use colored::Colorize;
            write!(f, "{}", string.bright_green())
        }
        #[cfg(not(feature = "native"))]
        {
            write!(f, "{}", string)
        }
    }
}

/// Utility method for debugging illegal actions
impl Game {
    pub fn debug_action_error(&self, action: &Action) -> String {
        let allowed_actions = self.legal();
        let state_info = format!(
            "Game state: dealer={}, ticker={}, street={}, pot={}",
            self.dealer,
            self.ticker,
            self.street(),
            self.pot
        );

        let actor_info = format!(
            "Actor (pos {}) state: {}, stack={}, stake={}",
            self.actor_idx(),
            self.actor_ref().state(),
            self.actor_ref().stack(),
            self.actor_ref().stake()
        );

        let effective = format!(
            "Effective stake: {}, to_call: {}, to_raise: {}, to_shove: {}",
            self.effective_stake(),
            self.to_call(),
            self.to_raise(),
            self.to_shove()
        );

        let is_info = format!(
            "Constraints: may_fold={}, may_call={}, may_check={}, may_raise={}, may_shove={}",
            self.may_fold(),
            self.may_call(),
            self.may_check(),
            self.may_raise(),
            self.may_shove()
        );

        format!(
            "Illegal action attempted: {}\nAllowed actions: {:?}\n{}\n{}\n{}\n{}",
            action, allowed_actions, state_info, actor_info, effective, is_info
        )
    }

    /// Finds the nearest allowed action to the provided action
    /// Returns None if there is no similar action allowed
    pub fn find_nearest_action(&self, action: &Action) -> Option<Action> {
        match action {
            Action::Raise(chips) => {
                // Shortcut: if raising is currently disallowed (the minimum
                // legal raise exceeds the player's remaining stack) but a
                // shove is still legal, map any raise attempt to an all-in.

                if !self.may_raise() && self.may_shove() {
                    return Some(Action::Shove(self.to_shove()));
                }

                // Suggest an alternative when the requested raise is outside
                // the legal range.  If the player attempts to raise *at least*
                // the maximum stack, treat it as a shove.  This spares the
                // client from having to distinguish between a very large raise
                // and an all-in.

                let min_raise = self.to_raise();
                let max_raise = self.to_shove() - 1;

                if *chips < min_raise {
                    // Too small → bump to minimum legal raise.
                    Some(Action::Raise(min_raise))
                } else if *chips >= self.to_shove() {
                    // Larger than stack → convert to shove if allowed.
                    if self.may_shove() {
                        Some(Action::Shove(self.to_shove()))
                    } else {
                        None
                    }
                } else if *chips > max_raise {
                    // Between (max_raise, to_shove) is not a legal raise size.
                    // Move to the largest legal raise (max_raise).
                    Some(Action::Raise(max_raise))
                } else {
                    // Inside legal range – keep amount.
                    Some(Action::Raise(*chips))
                }
            }
            Action::Shove(_chips) => {
                // Find nearest Shove action
                if self.may_shove() {
                    // Always use the actual stack size
                    return Some(Action::Shove(self.to_shove()));
                }
                None
            }
            // For other action types, no "close" alternative exists
            _ => None,
        }
    }
}

impl Game {
    /// Reset cards at a specific position - helper for Recall to avoid card conflicts
    pub fn reset_cards_at(mut self, position: usize, hole: Hole) -> Self {
        if position < self.seats.len() {
            self.seats[position].reset_cards(hole);
        }
        self
    }

    /// Hole cards of the requested player, as dealt at the beginning of the hand.
    /// The index is modulo `N` and the returned value is *private* to that seat.
    pub fn hole_cards(&self, position: usize) -> Hole {
        self.seats
            .get(position % self.n())
            .map(|seat| seat.cards())
            .expect("player index in bounds")
    }
}

impl Game {
    /// Build an Observation from the hero's pocket cards (given by `hero_idx`)
    /// and the public board, independent of whose turn it is.  This is needed
    /// for blueprint look-ups where the card information (present) must not be
    /// affected by betting order.
    pub fn sweat_for(&self, hero_idx: usize) -> Observation {
        use crate::cards::hand::Hand;
        Observation::from((
            Hand::from(self.seats[hero_idx % self.n()].cards()),
            Hand::from(self.board()),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_root() {
        let game = Game::root();
        assert!(game.ticker != game.dealer);
        assert!(game.board().street() == Street::Pref);
        assert!(game.actor().state() == State::Betting);
        assert!(game.pot() == Game::sblind() + Game::bblind());
    }

    #[test]
    fn everyone_folds_pref() {
        let game = Game::root();

        if crate::N == 2 {
            // Heads-up expectation – original assertions
            let game = game.apply(Action::Fold);
            assert!(game.is_everyone_folding());
            assert!(game.is_everyone_alright());
            assert!(!game.is_everyone_calling());
            assert!(game.must_deal());
            assert!(game.must_stop());
        } else {
            // Multi-way: SB call, BB check, Dealer check → ready to deal the flop

            // Pot should contain at least the small blind already
            assert!(game.pot() >= Game::sblind());
            assert!(game.street() == Street::Pref);

            // Actor is UTG (dealer) – they call the full big blind (2)
            let to_call = game.to_call();
            assert!(to_call > 0);
            let game = game.apply(Action::Call(to_call));

            // Actor is now the small blind; they still owe 1 chip and must call
            let to_call_sb = game.to_call();
            assert_eq!(to_call_sb, Game::bblind() - Game::sblind());
            let game = game.apply(Action::Call(to_call_sb));

            // Actor is the big blind; stakes are matched so they may check
            assert_eq!(game.to_call(), 0);
            let game = game.apply(Action::Check);

            // Everyone has acted and stakes are matched
            assert!(game.is_everyone_calling());
            assert!(game.must_deal());
            assert!(!game.must_stop());
        }
    }
    #[test]
    fn everyone_folds_flop() {
        if crate::N != 2 {
            // TODO: implement multi-way variant once flow differences are specified
            return;
        }
        let game = Game::root();
        let flop = game.deck().deal(Street::Pref);
        let game = game.apply(Action::Call(1));
        let game = game.apply(Action::Check);
        let game = game.apply(Action::Draw(flop));
        let game = game.apply(Action::Raise(10));
        let game = game.apply(Action::Fold);
        assert!(game.is_everyone_folding() == true);
        assert!(game.is_everyone_alright() == true); // fail
        assert!(game.is_everyone_calling() == false);
        assert!(game.must_deal() == true); // ambiguous
        assert!(game.must_stop() == true);
    }
    #[test]
    fn history_of_checks() {
        if crate::N != 2 {
            // TODO: implement multi-way variant once flow differences are specified
            return;
        }

        // Blinds
        let game = Game::root();
        assert!(game.board().street() == Street::Pref);
        assert!(game.pot() == 3);
        assert!(game.must_post() == false);
        assert!(game.must_stop() == false);
        assert!(game.must_deal() == false);
        assert!(game.is_everyone_alright() == false);
        assert!(game.is_everyone_calling() == false);
        assert!(game.is_everyone_touched() == false);
        assert!(game.is_everyone_matched() == false);

        // SmallB Preflop
        let game = game.apply(Action::Call(1));
        assert!(game.board().street() == Street::Pref);
        assert!(game.pot() == 4); //
        assert!(game.must_post() == false);
        assert!(game.must_stop() == false);
        assert!(game.must_deal() == false);
        assert!(game.is_everyone_alright() == false);
        assert!(game.is_everyone_calling() == false);
        assert!(game.is_everyone_touched() == false);
        assert!(game.is_everyone_matched() == true); //

        // Dealer Preflop
        let game = game.apply(Action::Check);
        assert!(game.board().street() == Street::Pref);
        assert!(game.pot() == 4);
        assert!(game.must_post() == false);
        assert!(game.must_stop() == false);
        assert!(game.must_deal() == true); //
        assert!(game.is_everyone_alright() == true); //
        assert!(game.is_everyone_calling() == true); //
        assert!(game.is_everyone_touched() == true); //
        assert!(game.is_everyone_matched() == true);

        // Flop
        let flop = game.deck().deal(game.board().street());
        let game = game.apply(Action::Draw(flop));
        assert!(game.board().street() == Street::Flop); //
        assert!(game.pot() == 4);
        assert!(game.must_post() == false);
        assert!(game.must_stop() == false);
        assert!(game.must_deal() == false); //
        assert!(game.is_everyone_alright() == false); //
        assert!(game.is_everyone_calling() == false); //
        assert!(game.is_everyone_touched() == false); //
        assert!(game.is_everyone_matched() == true);

        // SmallB Flop
        let game = game.apply(Action::Check);
        assert!(game.board().street() == Street::Flop);
        assert!(game.pot() == 4);
        assert!(game.must_post() == false);
        assert!(game.must_stop() == false);
        assert!(game.must_deal() == false);
        assert!(game.is_everyone_alright() == false);
        assert!(game.is_everyone_calling() == false);
        assert!(game.is_everyone_touched() == false);
        assert!(game.is_everyone_matched() == true);

        // Dealer Flop
        let game = game.apply(Action::Check);
        assert!(game.board().street() == Street::Flop);
        assert!(game.pot() == 4);
        assert!(game.must_post() == false);
        assert!(game.must_stop() == false);
        assert!(game.must_deal() == true); //
        assert!(game.is_everyone_alright() == true); //
        assert!(game.is_everyone_calling() == true); //
        assert!(game.is_everyone_touched() == true); //
        assert!(game.is_everyone_matched() == true);

        // Turn
        let turn = game.deck().deal(game.board().street());
        let game = game.apply(Action::Draw(turn));
        assert!(game.board().street() == Street::Turn);
        assert!(game.pot() == 4);
        assert!(game.must_post() == false);
        assert!(game.must_stop() == false);
        assert!(game.must_deal() == false); //
        assert!(game.is_everyone_alright() == false); //
        assert!(game.is_everyone_calling() == false); //
        assert!(game.is_everyone_touched() == false); //
        assert!(game.is_everyone_matched() == true);

        // SmallB Turn
        let game = game.apply(Action::Check);
        assert!(game.board().street() == Street::Turn);
        assert!(game.pot() == 4);
        assert!(game.must_post() == false);
        assert!(game.must_stop() == false);
        assert!(game.must_deal() == false);
        assert!(game.is_everyone_alright() == false);
        assert!(game.is_everyone_calling() == false);
        assert!(game.is_everyone_touched() == false);
        assert!(game.is_everyone_matched() == true);

        // Dealer Turn
        let game = game.apply(Action::Raise(4));
        assert!(game.board().street() == Street::Turn);
        assert!(game.pot() == 8);
        assert!(game.must_post() == false);
        assert!(game.must_stop() == false);
        assert!(game.must_deal() == false);
        assert!(game.is_everyone_alright() == false);
        assert!(game.is_everyone_calling() == false);
        assert!(game.is_everyone_touched() == true); //
        assert!(game.is_everyone_matched() == false); //

        // SmallB Turn
        let game = game.apply(Action::Call(4));
        assert!(game.board().street() == Street::Turn);
        assert!(game.pot() == 12); //
        assert!(game.must_post() == false);
        assert!(game.must_stop() == false);
        assert!(game.must_deal() == true); //
        assert!(game.is_everyone_alright() == true); //
        assert!(game.is_everyone_calling() == true); //
        assert!(game.is_everyone_touched() == true);
        assert!(game.is_everyone_matched() == true);

        // River
        let rive = game.deck().deal(game.board().street());
        let game = game.apply(Action::Draw(rive));
        assert!(game.board().street() == Street::Rive); //
        assert!(game.pot() == 12);
        assert!(game.must_post() == false);
        assert!(game.must_stop() == false);
        assert!(game.must_deal() == false); //
        assert!(game.is_everyone_alright() == false); //
        assert!(game.is_everyone_calling() == false); //
        assert!(game.is_everyone_touched() == false); //
        assert!(game.is_everyone_matched() == true); //

        // SmallB River
        let game = game.apply(Action::Check);
        assert!(game.board().street() == Street::Rive);
        assert!(game.pot() == 12);
        assert!(game.must_post() == false);
        assert!(game.must_stop() == false);
        assert!(game.must_deal() == false);
        assert!(game.is_everyone_alright() == false);
        assert!(game.is_everyone_calling() == false);
        assert!(game.is_everyone_touched() == false);
        assert!(game.is_everyone_matched() == true);

        // Dealer River
        let game = game.apply(Action::Check);
        assert!(game.board().street() == Street::Rive);
        assert!(game.pot() == 12);
        assert!(game.must_post() == false);
        assert!(game.must_stop() == true); //
        assert!(game.must_deal() == false);
        assert!(game.is_everyone_alright() == true); //
        assert!(game.is_everyone_calling() == true); //
        assert!(game.is_everyone_touched() == true); //
        assert!(game.is_everyone_matched() == true); //
    }
}
