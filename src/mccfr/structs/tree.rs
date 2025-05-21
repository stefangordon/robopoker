use super::infoset::InfoSet;
use super::node::Node;
use crate::mccfr::traits::edge::Edge;
use crate::mccfr::traits::game::Game;
use crate::mccfr::traits::info::Info;
use crate::mccfr::traits::turn::Turn;
use crate::mccfr::types::branch::Branch;
use petgraph::graph::NodeIndex;
use rustc_hash::FxHashMap;

/// the tree is pre-implemented. it is a wrapper around
/// a petgraph::graph::DiGraph. at each vertex, we store a
/// tuple of the fully abstracted Game and Info.
///
/// we assume that we are generated recursively from Encoder and Profile.
/// together, these traits enable "exploring the game space" up to the
/// rules of the game, i.e. implementation of T, E, G, I, Encoder, Profile.
#[derive(Debug)]
pub struct Tree<T, E, G, I>
where
    T: Turn,
    E: Edge,
    G: Game<E = E, T = T>,
    I: Info<E = E, T = T>,
{
    graph: petgraph::graph::DiGraph<(G, u32), E>,
    arena: Vec<I>,
    map: FxHashMap<I, u32>,
    danny: std::marker::PhantomData<T>,
}

impl<T, E, G, I> Tree<T, E, G, I>
where
    T: Turn,
    E: Edge,
    G: Game<E = E, T = T>,
    I: Info<E = E, T = T>,
{
    /// get all Nodes in the Tree
    pub fn all(&self) -> impl Iterator<Item = Node<T, E, G, I>> {
        self.graph.node_indices().map(|n| self.at(n))
    }
    /// get a Node by index
    pub fn at(&self, index: petgraph::graph::NodeIndex) -> Node<T, E, G, I> {
        Node::from(index, &self.graph, &self.arena)
    }
    /// seed a Tree by giving an (Info, Game) and getting a Node
    pub fn seed(&mut self, info: I, seed: G) -> Node<T, E, G, I> {
        let id = self.intern(info);
        let seed = self.graph.add_node((seed, id));
        self.at(seed)
    }
    /// extend a Tree by giving a Leaf and getting a Node
    pub fn grow(&mut self, info: I, leaf: Branch<E, G>) -> Node<T, E, G, I> {
        let id = self.intern(info);
        let tail = self.graph.add_node((leaf.1, id));
        let edge = self.graph.add_edge(leaf.2, tail, leaf.0);
        assert!(edge.index() == tail.index() - 1);
        self.at(tail)
    }
    /// group non-leaf Nodes by Info into InfoSets
    pub fn partition(self) -> FxHashMap<I, InfoSet<T, E, G, I>> {
        let tree = std::sync::Arc::new(self);
        let mut info: FxHashMap<I, InfoSet<T, E, G, I>> = FxHashMap::default();
        for node in tree.all().filter(|n| n.children().len() > 0) {
            info.entry(node.info().clone())
                .or_insert_with(|| InfoSet::from(tree.clone()))
                .push(node.index());
        }
        info
    }

    /// display the Tree in a human-readable format
    /// be careful because it's really big and recursive
    fn show(&self, f: &mut std::fmt::Formatter, x: NodeIndex, prefix: &str) -> std::fmt::Result {
        if x == NodeIndex::new(0) {
            writeln!(f, "\nROOT   {:?}", self.at(x).info())?;
        }
        let children = self
            .graph
            .neighbors_directed(x, petgraph::Outgoing)
            .collect::<Vec<_>>();
        let n = children.len();
        for (i, child) in children.into_iter().rev().enumerate() {
            let last = i == n - 1;
            let gaps = if last { "    " } else { "│   " };
            let stem = if last { "└" } else { "├" };
            let node = self.at(child);
            let head = node.info();
            let edge = self
                .graph
                .edge_weight(self.graph.find_edge(x, child).unwrap())
                .unwrap();
            writeln!(f, "{}{}──{:?} → {:?}", prefix, stem, edge, head)?;
            self.show(f, child, &format!("{}{}", prefix, gaps))?;
        }
        Ok(())
    }

    fn intern(&mut self, info: I) -> u32 {
        if let Some(&id) = self.map.get(&info) {
            id
        } else {
            let id = self.arena.len() as u32;
            self.arena.push(info);
            self.map.insert(info, id);
            id
        }
    }
}

impl<T, E, G, I> Default for Tree<T, E, G, I>
where
    T: Turn,
    E: Edge,
    G: Game<E = E, T = T>,
    I: Info<E = E, T = T>,
{
    fn default() -> Self {
        Self {
            graph: petgraph::graph::DiGraph::default(),
            arena: Vec::new(),
            map: FxHashMap::default(),
            danny: std::marker::PhantomData::<T>,
        }
    }
}

impl<T, E, G, I> std::fmt::Display for Tree<T, E, G, I>
where
    T: Turn,
    E: Edge,
    I: Info<E = E, T = T>,
    G: Game<E = E, T = T>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.show(f, NodeIndex::new(0), "")
    }
}
