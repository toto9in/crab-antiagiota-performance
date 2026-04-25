use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    routing::{get, post},
};
use serde::Serialize;

use crate::{payload::FraudRequest, state::AppState};

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/ready", get(ready))
        .route("/fraud-score", post(fraud_score))
        .with_state(state)
}

async fn ready() -> StatusCode {
    StatusCode::OK
}

#[derive(Serialize)]
struct FraudScoreResponse {
    approved: bool,
    fraud_score: f32,
}

async fn fraud_score(
    State(state): State<AppState>,
    Json(req): Json<FraudRequest>,
) -> Result<Json<FraudScoreResponse>, StatusCode> {
    let analysis = state.fraud_detector.analyze(&req);

    Ok(Json(FraudScoreResponse {
        approved: analysis.approved,
        fraud_score: analysis.fraud_score,
    }))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use axum::http::{Method, Request, StatusCode};
    use chrono::{TimeZone, Utc};
    use tower::util::ServiceExt;

    use super::router;
    use crate::{
        classifier::FraudDetector,
        dataset::ReferenceDataset,
        distance::DistanceEngine,
        payload::{Customer, FraudRequest, Merchant, Terminal, Transaction},
        state::AppState,
    };

    fn sample_request() -> FraudRequest {
        FraudRequest {
            id: "tx-1".into(),
            transaction: Transaction {
                amount: 41.12,
                installments: 2,
                requested_at: Utc.with_ymd_and_hms(2026, 3, 11, 18, 45, 53).unwrap(),
            },
            customer: Customer {
                avg_amount: 82.24,
                tx_count_24h: 3,
                known_merchants: vec!["MERC-003".into(), "MERC-016".into()],
            },
            merchant: Merchant {
                id: "MERC-016".into(),
                mcc: "5411".into(),
                avg_amount: 60.25,
            },
            terminal: Terminal {
                is_online: false,
                card_present: true,
                km_from_home: 29.2331036248,
            },
            last_transaction: None,
        }
    }

    #[tokio::test]
    async fn fraud_score_returns_success() {
        let dataset = Arc::new(ReferenceDataset::load_embedded().unwrap());
        let state = AppState::new(Arc::new(FraudDetector::new(
            dataset,
            DistanceEngine::scalar(),
        )));
        let app = router(state);
        let body = serde_json::to_vec(&sample_request()).unwrap();

        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/fraud-score")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }
}
