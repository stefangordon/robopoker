use crate::mccfr::traits::edge::Edge;
use crate::mccfr::traits::game::Game;
use crate::mccfr::traits::info::Info;
use crate::mccfr::traits::turn::Turn;
use crate::mccfr::types::branch::Branch;
use petgraph::graph::DiGraph;
use petgraph::graph::NodeIndex;

/// the node is pre-implemented. it is a wrapper around
/// a NodeIndex, and a thread-safe readonly reference
/// to the Tree in which it resides.
///
/// by only assuming the tree property of the underlying graph,
/// we can implement navigation methods recursively. all while being
/// fully generic over Turn Edge Game Info. just that they need to be
#[derive(Copy, Clone)]
pub struct Node<'tree, T, E, G, I>
where
    T: Turn,
    E: Edge,
    G: Game<E = E, T = T>,
    I: Info<E = E, T = T>,
{
    index: NodeIndex,
    graph: &'tree DiGraph<(G, u32), E>,
    arena: &'tree Vec<I>,
    danny: std::marker::PhantomData<T>,
}

impl<'tree, T, E, G, I> Node<'tree, T, E, G, I>
where
    T: Turn,
    E: Edge,
    G: Game<E = E, T = T>,
    I: Info<E = E, T = T>,
{
    pub fn from(index: NodeIndex, graph: &'tree DiGraph<(G, u32), E>, arena: &'tree Vec<I>) -> Self {
        Self {
            index,
            graph,
            arena,
            danny: std::marker::PhantomData::<T>,
        }
    }
    pub fn index(&self) -> NodeIndex {
        self.index
    }
    pub fn graph(&self) -> &'tree DiGraph<(G, u32), E> {
        self.graph
    }
    pub fn game(&self) -> &G {
        &self
            .graph()
            .node_weight(self.index())
            .expect("valid game index")
            .0
    }
    pub fn info(&self) -> &I {
        let id = self
            .graph()
            .node_weight(self.index())
            .expect("valid info index")
            .1 as usize;
        &self.arena[id]
    }
    pub fn at(&self, index: NodeIndex) -> Node<'tree, T, E, G, I> {
        Self {
            index,
            graph: self.graph(),
            arena: self.arena,
            danny: std::marker::PhantomData::<T>,
        }
    }

    pub fn up(&self) -> Option<(Node<'tree, T, E, G, I>, &'tree E)> {
        match (self.parent(), self.incoming()) {
            (Some(parent), Some(incoming)) => Some((parent, incoming)),
            (Some(_), _) => unreachable!("tree property violation"),
            (_, Some(_)) => unreachable!("tree property violation"),
            (None, None) => None,
        }
    }
    pub fn parent(&self) -> Option<Node<'tree, T, E, G, I>> {
        self.graph()
            .neighbors_directed(self.index(), petgraph::Direction::Incoming)
            .next()
            .map(|index| self.at(index))
    }
    pub fn incoming(&self) -> Option<&'tree E> {
        self.graph()
            .edges_directed(self.index(), petgraph::Direction::Incoming)
            .next()
            .map(|edge| edge.weight())
    }
    pub fn follow(&self, edge: &E) -> Option<Node<'tree, T, E, G, I>> {
        self.children()
            .iter()
            .find(|child| edge == child.incoming().unwrap())
            .map(|child| self.at(child.index()))
    }
    pub fn outgoing(&self) -> Vec<&'tree E> {
        self.graph()
            .edges_directed(self.index(), petgraph::Direction::Outgoing)
            .map(|edge| edge.weight())
            .collect()
    }
    pub fn children(&self) -> Vec<Node<'tree, T, E, G, I>> {
        self.graph()
            .neighbors_directed(self.index(), petgraph::Direction::Outgoing)
            .map(|index| self.at(index))
            .collect()
    }
    pub fn descendants(&self) -> Vec<Node<'tree, T, E, G, I>> {
        // Avoid recursion to prevent deep call stacks and repeated temporary vector
        // allocations.  We perform an explicit depth-first traversal using a stack.
        let mut stack = Vec::new();
        let mut leaves = Vec::new();

        stack.push(*self);

        while let Some(node) = stack.pop() {
            let children = node.children();
            if children.is_empty() {
                // Leaf reached – store it
                leaves.push(node);
            } else {
                // Continue traversal
                stack.extend(children);
            }
        }

        leaves
    }
    pub fn branches(&self) -> Vec<Branch<E, G>> {
        self.info()
            .choices()
            .into_iter()
            .map(|e| (e.clone(), self.game().apply(e), self.index()))
            .collect()
    }
}

/// Node naturally implements Iterator by recursing upward through its tree.
/// Each iteration yields a tuple of (Node, Edge) representing the parent node
/// and the edge taken to reach the current node. This allows traversing
/// from any node back to the root of the tree.
impl<'tree, T, E, G, I> Iterator for Node<'tree, T, E, G, I>
where
    T: Turn,
    E: Edge,
    G: Game<E = E, T = T>,
    I: Info<E = E, T = T>,
{
    type Item = (Self, E);
    fn next(&mut self) -> Option<Self::Item> {
        let (ref mut parent, edge) = self.up()?;
        std::mem::swap(self, parent);
        Some((self.clone(), edge.clone()))
    }
}

/// Debug + Display implementations, which will
/// treat a Node just as a combination of its
/// associated Info + its location in the tree
impl<'tree, T, E, G, I> std::fmt::Debug for Node<'tree, T, E, G, I>
where
    T: Turn,
    E: Edge,
    G: Game<E = E, T = T>,
    I: Info<E = E, T = T>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:?} ({}/{})",
            self.info(),
            self.index().index(),
            self.graph().node_count()
        )
    }
}

/// Eq implementation will assume that any two
/// Nodes being compared to one another belong
/// to the same tree/graph. such that, we only
/// care about comparing indices.
impl<'tree, T, E, G, I> PartialEq for Node<'tree, T, E, G, I>
where
    T: Turn,
    E: Edge,
    G: Game<E = E, T = T>,
    I: Info<E = E, T = T>,
{
    fn eq(&self, other: &Self) -> bool {
        self.index() == other.index() && std::ptr::eq(self.graph(), other.graph())
    }
}
impl<'tree, T, E, G, I> Eq for Node<'tree, T, E, G, I>
where
    T: Turn,
    E: Edge,
    G: Game<E = E, T = T>,
    I: Info<E = E, T = T>,
{
}
