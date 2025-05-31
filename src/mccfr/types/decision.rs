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
    serializer.serialize_str(&edge.to_string())
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

impl From<tokio_postgres::Row> for Decision {
    fn from(row: tokio_postgres::Row) -> Self {
        Self {
            edge: Edge::from(row.get::<_, i64>("edge") as u64),
            prob: Probability::from(row.get::<_, f32>("policy")),
        }
    }
} 