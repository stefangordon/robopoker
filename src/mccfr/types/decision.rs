use crate::gameplay::edge::Edge;
use crate::Probability;
use serde::Serialize;

#[derive(Serialize, Clone)]
pub struct Decision {
    #[serde(serialize_with = "serialize_edge")]
    pub edge: Edge,
    pub prob: Probability,
}

fn serialize_edge<S>(edge: &Edge, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    let edge_str = match edge {
        Edge::Draw => "?".to_string(),
        Edge::Fold => "F".to_string(),
        Edge::Check => "O".to_string(),
        Edge::Call => "*".to_string(),
        Edge::Shove => "!".to_string(),
        Edge::Raise(odds) => format!("{}:{}", odds.0, odds.1), // Use precise ratio format
    };
    serializer.serialize_str(&edge_str)
}

impl Decision {
    /// Get the edge for this decision
    pub fn edge(&self) -> Edge {
        self.edge
    }
    
    /// Get the probability/weight for this decision
    pub fn weight(&self) -> Probability {
        self.prob
    }
}

impl From<(Edge, Probability)> for Decision {
    fn from((edge, prob): (Edge, Probability)) -> Self {
        Self {
            edge,
            prob,
        }
    }
} 