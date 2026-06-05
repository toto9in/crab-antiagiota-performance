use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use std::sync::Arc;

use crate::quantizer::quantize_int16;
use crate::reference_store::ReferenceStore;
use crate::transaction::{FraudScoreRequest, FraudScoreResponse};
use crate::vectorizer::vectorize;

#[derive(Clone)]
pub struct AppState {
    references: Arc<ReferenceStore>,
}

impl AppState {
    pub fn new(references: ReferenceStore) -> Self {
        Self {
            references: Arc::new(references),
        }
    }
}

pub async fn ready() -> StatusCode {
    StatusCode::OK
}

pub async fn fraud_score(
    State(state): State<AppState>,
    Json(payload): Json<FraudScoreRequest>,
) -> Json<FraudScoreResponse> {
    let features = vectorize(&payload);
    let quantized = quantize_int16(&features);
    let fraud_score = state.references.fraud_score_for(&quantized);

    Json(FraudScoreResponse {
        approved: fraud_score < 0.6,
        fraud_score,
    })
}
