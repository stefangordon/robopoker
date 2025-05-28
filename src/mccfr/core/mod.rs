use crate::mccfr::types::decision::Decision;
use crate::mccfr::traits::{encoder::Encoder as EncoderTrait, profile::Profile as ProfileTrait};

/// Trait for plug-in convergence checking policies.
/// Implementations decide when the outer MCCFR loop should stop.
pub trait ConvergenceRule {
    /// Return `true` when the strategy has converged.
    fn has_converged(&mut self, old: &[Decision], new: &[Decision]) -> bool;
}

/// Never triggers – use for blueprint training.
#[derive(Default, Copy, Clone)]
pub struct NeverConverge;
impl ConvergenceRule for NeverConverge {
    #[inline]
    fn has_converged(&mut self, _old: &[Decision], _new: &[Decision]) -> bool {
        false
    }
}

/// Simple L1-distance threshold on the change vector.
#[derive(Copy, Clone)]
pub struct DeltaThreshold {
    pub eps: f32,
}


impl ConvergenceRule for DeltaThreshold {
    fn has_converged(&mut self, old: &[Decision], new: &[Decision]) -> bool {
        if old.len() != new.len() {
            return false;
        }
        let total_change: f32 = old
            .iter()
            .zip(new.iter())
            .map(|(a, b)| (a.weight() - b.weight()).abs())
            .sum();
        total_change < self.eps
    }
}

/// Generic MCCFR driver that owns an `Encoder`, a `Profile`, and a `ConvergenceRule`.
/// The concrete solve loops for blueprint training and sub-game re-solving become
/// thin wrappers that configure these three components.
pub struct Driver<P, Enc, Stop>
where
    P: ProfileTrait,
    Enc: EncoderTrait<T = P::T, E = P::E, G = P::G, I = P::I>,
    Stop: ConvergenceRule,
{
    pub profile: P,
    pub encoder: Enc,
    pub stop: Stop,
}

impl<P, Enc, Stop> Driver<P, Enc, Stop>
where
    P: ProfileTrait,
    Enc: EncoderTrait<T = P::T, E = P::E, G = P::G, I = P::I>,
    Stop: ConvergenceRule,
{
    pub fn new(profile: P, encoder: Enc, stop: Stop) -> Self {
        Self { profile, encoder, stop }
    }

    /// Generic solve loop.  Returns the root strategy after `max_iter` epochs or
    /// earlier if the `stop` rule fires after `min_iter` epochs.
    pub fn solve<G>(&mut self, _root_game: &P::G, _min_iter: usize, _max_iter: usize) -> Vec<Decision> {
        unimplemented!("Generic driver solve loop not yet integrated – to be implemented in follow-up commits.")
    }

    // --- helper methods will be re-added when the driver integration is completed ---
}