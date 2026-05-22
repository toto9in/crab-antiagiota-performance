use axum::Json;
use axum::http::StatusCode;

use crate::quantizer::quantize_int8;
use crate::transaction::{FraudScoreRequest, FraudScoreResponse};
use crate::vectorizer::vectorize;

pub async fn ready() -> StatusCode {
    StatusCode::OK
}

pub async fn fraud_score(Json(payload): Json<FraudScoreRequest>) -> Json<FraudScoreResponse> {
    let features = vectorize(&payload);
    let _quantized = quantize_int8(&features);

    Json(FraudScoreResponse {
        approved: false,
        fraud_score: 0.5,
    })
}
