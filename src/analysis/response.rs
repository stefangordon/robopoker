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
    pub edge: String,
    pub prob: Probability,
}

impl Decision {
    /// Get the edge for this decision
    pub fn edge(&self) -> Edge {
        // Parse the edge string back to Edge
        // This is a bit hacky but works for now
        match self.edge.as_str() {
            "?" => Edge::Draw,
            "F" => Edge::Fold,
            "O" => Edge::Check,
            "*" => Edge::Call,
            "!" => Edge::Shove,
            s => {
                // Handle raise format like "3:2"
                if let Some((num, den)) = s.split_once(':') {
                    if let (Ok(n), Ok(d)) = (num.parse::<i16>(), den.parse::<i16>()) {
                        Edge::Raise(crate::gameplay::odds::Odds(n, d))
                    } else {
                        Edge::Check // Default fallback
                    }
                } else {
                    Edge::Check // Default fallback
                }
            }
        }
    }
    
    /// Get the probability/weight for this decision
    pub fn weight(&self) -> Probability {
        self.prob
    }
}

impl From<(Edge, Probability)> for Decision {
    fn from((edge, prob): (Edge, Probability)) -> Self {
        Self {
            edge: edge.to_string(),
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
            edge: Edge::from(row.get::<_, i64>("edge") as u64).to_string(),
            prob: Probability::from(row.get::<_, f32>("policy")),
        }
    }
}
