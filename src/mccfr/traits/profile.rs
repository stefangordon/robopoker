use super::edge::Edge;
use super::game::Game;
use super::info::Info;
use super::turn::Turn;
use crate::mccfr::structs::infoset::InfoSet;
use crate::mccfr::structs::node::Node;
use crate::mccfr::types::branch::Branch;
use crate::mccfr::types::policy::Policy;

/// The `Profile` trait represents a strategy profile in an extensive-form game, implementing core
/// functionality for Counterfactual Regret Minimization (CFR).
///
/// A strategy profile maintains and updates:
/// - Accumulated regrets for each information set and action
/// - Accumulated weighted average strategies (policies) over time
/// - Current iteration/epoch tracking
///
/// # Key Concepts
///
/// ## Strategy Computation
/// - `policy_vector`: Computes immediate strategy distribution using regret-matching
/// - `policy`: Calculates immediate strategy from accumulated regrets
/// - `advice`: Provides long-run average strategy (Nash equilibrium approximation)
///
/// ## Reach Probabilities
/// - `expected_reach`: Probability of reaching a node following the current strategy
/// - `cfactual_reach`: Counterfactual reach probability (excluding player's own actions)
/// - `relative_reach`: Conditional probability of reaching a leaf from a given node
///
/// ## Utility and Regret
/// - `regret_vector`: Computes counterfactual regret for all actions in an information set
/// - `info_gain`: Immediate regret for not an action in an information set
/// - `node_gain`: Immediate regret for not an action at a specific node
///
/// ## Sampling
/// - `sample`: Implements various sampling schemes (e.g., external, targeted, uniform)
/// - `rng`: Provides deterministic random number generation for consistent sampling
///
/// # Implementation Notes
///
/// Implementors must provide:
/// - `increment`: Update epoch/iteration counter
/// - `walker`: Current player's turn
/// - `epochs`: Number of iterations completed
/// - `weight`: Access to accumulated action weights/policies
/// - `regret`: Access to accumulated regrets
/// - `sample`: Custom sampling strategy
///
/// The trait provides automatic implementations for strategy computation, reach probabilities,
/// and utility calculations based on these core methods.
pub trait Profile {
    type T: Turn;
    type E: Edge;
    type G: Game<E = Self::E, T = Self::T>;
    type I: Info<E = Self::E, T = Self::T>;

    // unimplemented

    /// increment epoch
    fn increment(&mut self);
    /// who's turn is it?
    fn walker(&self) -> Self::T;
    /// how many iterations
    fn epochs(&self) -> usize;
    /// lookup accumulated policy for this information
    fn sum_policy(&self, info: &Self::I, edge: &Self::E) -> crate::Probability;
    /// lookup accumulated regret for this information
    fn sum_regret(&self, info: &Self::I, edge: &Self::E) -> crate::Utility;

    // exploration calculations

    /// topology-based sampling. i.e. external, probing, targeted, uniform, etc.
    ///
    /// this default implementation is opinionated about using
    /// external average discounted sampling
    /// - external: only the current traverser's actions are fully explored
    /// - average: the accumulated policy values are used to weight samples
    /// - discounted: a discounting schedule can adapt sensitiviy
    ///
    /// For vanilla CFR, no-op because we sample all actions
    fn explore(
        &self,
        node: &Node<Self::T, Self::E, Self::G, Self::I>,
        branches: Vec<Branch<Self::E, Self::G>>,
    ) -> Vec<Branch<Self::E, Self::G>> {
        let n = branches.len();
        let p = node.game().turn();
        let walker = self.walker();
        let chance = Self::T::chance();
        match (n, p) {
            (0, _) => branches,
            (_, p) if p == walker => branches,
            (_, p) if p == chance => self.explore_any(node, branches),
            (_, p) if p != walker => self.explore_one(node, branches),
            _ => panic!("at the disco"),
        }
    }
    /// uniform sampling of available branches
    fn explore_any(
        &self,
        node: &Node<Self::T, Self::E, Self::G, Self::I>,
        branches: Vec<Branch<Self::E, Self::G>>,
    ) -> Vec<Branch<Self::E, Self::G>> {
        use rand::Rng;
        use std::ops::Not;
        assert!(branches.is_empty().not());
        let n = branches.len();
        let mut choices = branches;
        let ref mut rng = self.rng(node.info());
        let choice = rng.gen_range(0..n);
        let chosen = choices.remove(choice);
        vec![chosen]
    }
    /// profile-weighted by ACCUMULATED WEIGHTED AVERAGE
    /// policy policy values. discounting is encapsulated
    /// by self.discount(_)
    fn explore_one(
        &self,
        node: &Node<Self::T, Self::E, Self::G, Self::I>,
        branches: Vec<Branch<Self::E, Self::G>>,
    ) -> Vec<Branch<Self::E, Self::G>> {
        use rand::distributions::WeightedIndex;
        use rand::prelude::Distribution;
        let ref info = node.info();
        let ref mut rng = self.rng(info);
        let mut choices = branches;
        let policy = choices
            .iter()
            .map(|(edge, _, _)| self.sample(info, edge))
            .collect::<Vec<_>>();
        let choice = WeightedIndex::new(policy)
            .expect("at least one policy > 0")
            .sample(rng);
        let chosen = choices.remove(choice);
        vec![chosen]
    }

    // update vector calculations

    /// Using our current strategy Profile,
    /// compute the regret vector
    /// by calculating the marginal Utitlity
    /// missed out on for not having followed
    /// every walkable Edge at this Infoset/Node/Bucket
    fn regret_vector(
        &self,
        infoset: &InfoSet<Self::T, Self::E, Self::G, Self::I>,
    ) -> Policy<Self::E> {
        // Collect the head nodes of this infoset only once.
        let roots = infoset.span();

        // Pre-compute the expected value for each root. This value is identical
        // for every edge we are about to iterate over, so caching avoids an
        // expensive `expected_value` call per (root, edge) pair.
        let expected_by_root: Vec<crate::Utility> = roots
            .iter()
            .map(|root| self.expected_value(root))
            .collect();

        // Iterate over the available choices exactly once, producing the regret
        // for each edge. We accumulate (cfactual - expected) across all roots.
        infoset
            .info()
            .choices()
            .into_iter()
            .map(|edge| {
                let mut regret_sum = 0.0;
                for (idx, root) in roots.iter().enumerate() {
                    let cfactual = self.cfactual_value(root, &edge);
                    let expected = expected_by_root[idx];

                    debug_assert!(!cfactual.is_nan(), "cfactual_value produced NaN for edge {:?}", edge);
                    debug_assert!(!expected.is_nan(), "expected_value produced NaN");
                    debug_assert!(!cfactual.is_infinite(), "cfactual_value produced infinity for edge {:?}", edge);
                    debug_assert!(!expected.is_infinite(), "expected_value produced infinity");

                    regret_sum += cfactual - expected;
                }
                debug_assert!(!regret_sum.is_nan(), "regret_sum became NaN for edge {:?}", edge);
                debug_assert!(!regret_sum.is_infinite(), "regret_sum became infinite for edge {:?}", edge);
                (edge, regret_sum)
            })
            .collect::<Policy<Self::E>>()
    }
    /// calculate immediate policy distribution from current regrets, ignoring historical weighted policies.
    /// this uses regret matching, which converts regret values into probabilities by:
    /// 1. taking the positive portion of each regret (max with 0)
    /// 2. normalizing these values to sum to 1.0 to form a valid probability distribution
    /// this ensures actions with higher regret are chosen more frequently to minimize future regret.
    fn policy_vector(
        &self,
        infoset: &InfoSet<Self::T, Self::E, Self::G, Self::I>,
    ) -> Policy<Self::E> {
        let info = infoset.info();
        let regrets = info
            .choices()
            .into_iter()
            .map(|e| (e, self.sum_regret(&info, &e)))
            .map(|(a, r)| (a, r.max(crate::POLICY_MIN)))
            .collect::<Policy<Self::E>>();
        let denominator = regrets
            .iter()
            .map(|(_, r)| r)
            .inspect(|r| assert!(**r >= 0.))
            .sum::<crate::Utility>();
        let policy = regrets
            .into_iter()
            .map(|(a, r)| (a, r / denominator))
            .inspect(|(_, p)| assert!(*p >= 0.))
            .inspect(|(_, p)| assert!(*p <= 1.))
            .collect::<Policy<Self::E>>();
        policy
    }

    // strategy calculations

    /// calculate IMMEDIATE weighted average decision
    /// strategy for this information.
    /// i.e. policy from accumulated REGRET values
    fn policy(&self, info: &Self::I, edge: &Self::E) -> crate::Probability {
        let numer = self.sum_regret(info, edge).max(crate::POLICY_MIN);
        let denom = info
            .choices()
            .iter()
            .map(|e| self.sum_regret(info, e))
            .inspect(|r| assert!(!r.is_nan(), "regret value is NaN for edge in policy calculation"))
            .inspect(|r| assert!(!r.is_infinite(), "regret value is infinite for edge in policy calculation"))
            .map(|r| r.max(crate::POLICY_MIN))
            .sum::<crate::Utility>();

        debug_assert!(denom > 0.0, "denominator is zero or negative in policy calculation");
        let result = numer / denom;
        debug_assert!(!result.is_nan(), "policy calculation produced NaN: {} / {}", numer, denom);
        debug_assert!(result >= 0.0 && result <= 1.0, "policy value out of bounds: {}", result);

        result
    }
    /// calculate the HISTORICAL WEIGHTED AVERAGE decision
    /// strategy for this information.
    /// i.e. policy from accumulated POLICY values
    fn advice(&self, info: &Self::I, edge: &Self::E) -> crate::Probability {
        let numer = self.sum_policy(info, edge).max(crate::POLICY_MIN);
        let denom = info
            .choices()
            .iter()
            .map(|e| self.sum_policy(info, e))
            .inspect(|r| assert!(!r.is_nan()))
            .inspect(|r| assert!(!r.is_infinite()))
            .inspect(|r| assert!(*r >= 0.))
            .map(|r| r.max(crate::POLICY_MIN))
            .sum::<crate::Probability>();
        numer / denom
    }
    /// In Monte Carlo CFR variants, we sample actions according to a sampling strategy q(a).
    /// This function computes q(a) for a given action in an infoset, which is used for importance sampling.
    /// The sampling probability is based on the action weights, temperature, inertia, and exploration parameters.
    /// The formula is: q(a) = max(exploration, (inertia + temperature * weight(a)) / (inertia + sum(weights)))
    fn sample(&self, info: &Self::I, edge: &Self::E) -> crate::Probability {
        let numer = self.sum_policy(info, edge).max(crate::POLICY_MIN);
        let denom = info
            .choices()
            .iter()
            .map(|e| self.sum_policy(info, e))
            .inspect(|r| assert!(!r.is_nan()))
            .inspect(|r| assert!(!r.is_infinite()))
            .inspect(|r| assert!(*r >= 0.))
            .map(|r| r.max(crate::POLICY_MIN))
            .sum::<crate::Probability>();
        let denom = self.activation() + denom;
        let numer = self.activation() + numer * self.threshold();
        (numer / denom).max(self.exploration())
    }

    // reach calculations

    /// at the immediate location of this Node,
    /// what is the Probability of transitioning via this Edge?
    fn outgoing_reach(
        &self,
        node: &Node<Self::T, Self::E, Self::G, Self::I>,
        edge: &Self::E,
    ) -> crate::Probability {
        self.policy(node.info(), edge)
    }
    /// Conditional on being in a given Infoset,
    /// what is the Probability of
    /// visiting this particular leaf Node,
    /// assuming we all follow the distribution offered by Profile?
    fn relative_reach(
        &self,
        root: &Node<Self::T, Self::E, Self::G, Self::I>,
        leaf: &Node<Self::T, Self::E, Self::G, Self::I>,
    ) -> crate::Probability {
        leaf.into_iter()
            .take_while(|(parent, _)| parent != root) // parent.index() > root.index()
            .map(|(parent, incoming)| self.outgoing_reach(&parent, &incoming))
            .product::<crate::Probability>()
    }
    /// If we were to play by the Profile,
    /// up to this Node in the Tree,
    /// then what is the probability of visiting this Node?
    fn expected_reach(
        &self,
        root: &Node<Self::T, Self::E, Self::G, Self::I>,
    ) -> crate::Probability {
        root.into_iter()
            .map(|(parent, incoming)| self.outgoing_reach(&parent, &incoming))
            .product::<crate::Probability>()
    }
    /// If, counterfactually, we had played toward this infoset,
    /// then what would be the Probability of us being in this infoset?
    /// i.e. assuming our opponents played according to distributions from Profile, but we did not.
    ///
    /// This function also serves as a form of importance sampling.
    /// MCCFR requires we adjust our reach in counterfactual
    /// regret calculation to account for the under- and over-sampling
    /// of regret across different Infosets.
    fn cfactual_reach(
        &self,
        root: &Node<Self::T, Self::E, Self::G, Self::I>,
    ) -> crate::Probability {
        root.into_iter()
            .filter(|(parent, _)| self.walker() != parent.game().turn())
            .map(|(parent, incoming)| self.outgoing_reach(&parent, &incoming))
            .product::<crate::Probability>()
    }
    /// In Monte Carlo CFR variants, we sample actions according to some
    /// sampling strategy q(a) (possibly in place of the current policy p(a)).
    /// To correct for this bias, we multiply regrets by p(a)/q(a).
    /// This function returns q(a), the probability that we sampled
    /// the actions leading to this node under our sampling scheme.
    ///
    /// For vanilla CFR, q(a) = 1.0 since we explore all actions.
    fn sampling_reach(
        &self,
        leaf: &Node<Self::T, Self::E, Self::G, Self::I>,
    ) -> crate::Probability {
        let result = leaf.into_iter()
            .filter(|(parent, _)| self.walker() != parent.game().turn())
            .map(|(parent, incoming)| self.sample(parent.info(), &incoming))
            .product::<crate::Probability>()
            .max(crate::POLICY_MIN); // Prevent underflow to zero

        // In production builds, log warnings instead of panicking
        #[cfg(not(debug_assertions))]
        if result <= crate::POLICY_MIN {
            log::warn!("sampling_reach underflowed to minimum value, this may indicate numerical instability after {} epochs", self.epochs());
        }

        result
    }

    // utility calculations

    /// relative to the player at the root Node of this Infoset,
    /// what is the Utility of this leaf Node?
    fn relative_value(
        &self,
        root: &Node<Self::T, Self::E, Self::G, Self::I>,
        leaf: &Node<Self::T, Self::E, Self::G, Self::I>,
    ) -> crate::Utility {
        let reach = self.relative_reach(root, leaf);
        let payoff = leaf.game().payoff(root.game().turn());
        let sampling = self.sampling_reach(leaf);

        debug_assert!(!reach.is_nan(), "relative_reach produced NaN");
        debug_assert!(!payoff.is_nan(), "payoff produced NaN");
        debug_assert!(!sampling.is_nan(), "sampling_reach produced NaN");
        debug_assert!(sampling > 0.0, "sampling_reach is zero or negative: {}", sampling);

        // Prevent numeric overflow that can create ±Inf and later NaNs
        let mut result = reach * payoff / sampling;

        // Additional safety check for extreme values before they become infinite
        if result.abs() > crate::REGRET_MAX {
            result = result.signum() * crate::REGRET_MAX;
        }

        if !result.is_finite() {
            // Clamp to REGRET_MAX with correct sign to keep training stable
            result = result.signum() * crate::REGRET_MAX;
        }

        debug_assert!(!result.is_nan(), "relative_value produced NaN after clamp");
        debug_assert!(!result.is_infinite(), "relative_value still infinite after clamp");

        result
    }
    /// Assuming we start at root Node,
    /// and that we sample the Tree according to Profile,
    /// how much Utility do we expect upon
    /// visiting this Node?
    fn expected_value(&self, root: &Node<Self::T, Self::E, Self::G, Self::I>) -> crate::Utility {
        assert!(self.walker() == root.game().turn());
        self.expected_reach(root)
            * root
                .descendants()
                .iter()
                .map(|leaf| self.relative_value(root, leaf))
                .sum::<crate::Utility>()
    }
    /// If, counterfactually,
    /// we had intended to get ourselves in this infoset,
    /// then what would be the expected Utility of this leaf?
    fn cfactual_value(
        &self,
        root: &Node<Self::T, Self::E, Self::G, Self::I>,
        edge: &Self::E,
    ) -> crate::Utility {
        assert!(self.walker() == root.game().turn());
        self.cfactual_reach(root)
            * root
                .follow(edge)
                .expect("edge belongs to outgoing branches")
                .descendants()
                .iter()
                .map(|leaf| self.relative_value(root, leaf))
                .sum::<crate::Utility>()
    }

    // counterfactual gain calculations

    /// Conditional on being in this Infoset,
    /// distributed across all its head Nodes,
    /// with paths weighted according to our Profile:
    /// if we follow this Edge 100% of the time,
    /// what is the expected marginal increase in Utility?
    fn info_gain(
        &self,
        info: &InfoSet<Self::T, Self::E, Self::G, Self::I>,
        edge: &Self::E,
    ) -> crate::Utility {
        info.span()
            .iter()
            .map(|root| self.node_gain(root, edge))
            .inspect(|r| assert!(!r.is_nan()))
            .inspect(|r| assert!(!r.is_infinite()))
            .sum::<crate::Utility>()
    }
    /// Using our current strategy Profile, how much regret
    /// would we gain by following this Edge at this Node?
    fn node_gain(
        &self,
        root: &Node<Self::T, Self::E, Self::G, Self::I>,
        edge: &Self::E,
    ) -> crate::Utility {
        assert!(self.walker() == root.game().turn());
        let cfactual = self.cfactual_value(root, edge);
        let expected = self.expected_value(root);
        cfactual - expected
    }

    // deterministic sampling

    /// deterministically sampling the same Edge for the same Infoset
    /// requries decision-making to be Info-level
    fn rng(&self, info: &Self::I) -> rand::rngs::SmallRng {
        use rand::SeedableRng;
        use std::collections::hash_map::DefaultHasher;
        use std::hash::Hash;
        use std::hash::Hasher;
        let ref mut hasher = DefaultHasher::new();
        self.epochs().hash(hasher);
        info.hash(hasher);
        rand::rngs::SmallRng::seed_from_u64(hasher.finish())
    }

    // constant fns

    /// Tau (τ) - temperature parameter that controls sampling greediness.
    /// Set to 0.5 to make sampling more focused on promising actions while
    /// still maintaining some exploration.
    fn threshold(&self) -> crate::Entropy {
        crate::SAMPLING_THRESHOLD
    }
    /// Beta (β) - inertia parameter that stabilizes strategy updates by weighting
    /// historical policies. Set to 0.5 to balance between stability and adaptiveness.
    fn activation(&self) -> crate::Energy {
        crate::SAMPLING_ACTIVATION
    }
    /// Epsilon (ε) - exploration parameter that ensures minimum sampling probability
    /// for each action to maintain exploration. Set to 0.01 based on empirical testing
    /// which showed better convergence compared to higher values.
    fn exploration(&self) -> crate::Probability {
        crate::SAMPLING_EXPLORATION
    }
}
