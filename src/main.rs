use std::env;

use axum::{Router, routing::{get, post}};
use crab_antiagiota_performance::api;
use tokio::net::UnixListener;

#[tokio::main]
async fn main() {
    let socket_path = env::var("SOCKET_PATH").unwrap_or_else(|_| "/tmp/api.sock".into());

    let _ = std::fs::remove_file(&socket_path);

    let listener = UnixListener::bind(&socket_path).expect("bind unix socket");

    let app = Router::new()
        .route("/ready", get(api::ready))
        .route("/fraud-score", post(api::fraud_score));

    axum::serve(listener, app).await.expect("server error");
}
