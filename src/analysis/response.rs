use crate::cards::observation::Observation;
use crate::clustering::abstraction::Abstraction;
use crate::gameplay::edge::Edge;
use crate::Probability;
use serde::Serialize;

#[derive(Serialize)]
pub struct Sample {
    pub obs: String,
    pub abs: String,
    pub equity: f32,
    pub density: f32,
    pub distance: f32,
}

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

impl From<tokio_postgres::Row> for Sample {
    fn from(row: tokio_postgres::Row) -> Self {
        Self {
            obs: Observation::from(row.get::<_, i64>(0)).equivalent(),
            abs: Abstraction::from(row.get::<_, i64>(1)).to_string(),
            equity: row.get::<_, f32>(2).into(),
            density: row.get::<_, f32>(3).into(),
            distance: row.get::<_, f32>(4).into(),
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
