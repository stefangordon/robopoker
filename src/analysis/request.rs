use serde::Deserialize;

#[derive(Deserialize)]
pub struct SetStreets {
    pub street: String,
}

#[derive(Deserialize)]
pub struct ReplaceObs {
    pub obs: String,
}

#[derive(Deserialize)]
pub struct RowWrtObs {
    pub obs: String,
}

#[derive(Deserialize)]
pub struct ReplaceAbs {
    pub wrt: String,
}

#[derive(Deserialize)]
pub struct ReplaceRow {
    pub wrt: String,
    pub obs: String,
}

#[derive(Deserialize)]
pub struct ReplaceOne {
    pub wrt: String,
    pub abs: String,
}

#[derive(Deserialize)]
pub struct ReplaceAll {
    pub wrt: String,
    pub neighbors: Vec<String>,
}

#[derive(Deserialize)]
pub struct ObsHist {
    pub obs: String,
}

#[derive(Deserialize)]
pub struct AbsHist {
    pub abs: String,
}

#[derive(Deserialize)]
pub struct GetPolicy {
    pub hero: String,
    pub seen: String,
    pub path: Vec<String>,
}
