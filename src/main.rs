use std::env;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;

use axum::{
    Router,
    routing::{get, post},
};
use crab_antiagiota_performance::{api, reference_store::ReferenceStore};
use tokio::net::UnixListener;

fn main() {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .max_blocking_threads(2)
        .build()
        .unwrap()
        .block_on(run());
}

async fn run() {
    let socket_path = env::var("SOCKET_PATH").unwrap_or_else(|_| "/tmp/api.sock".into());

    let _ = std::fs::remove_file(&socket_path);

    let listener = UnixListener::bind(&socket_path).expect("bind unix socket");
    std::fs::set_permissions(&socket_path, PermissionsExt::from_mode(0o777))
        .expect("set socket permissions");

    // Load the prebuilt index. The server only starts serving if this succeeds;
    // a missing/corrupt .bin panics here, before any route is mounted.
    let index_path = env::var("INDEX_PATH").unwrap_or_else(|_| "index.bin".into());
    let references = ReferenceStore::from_bin_path(Path::new(&index_path))
        .unwrap_or_else(|err| panic!("load index from {index_path}: {err}"));
    let state = api::AppState::new(references);

    let app = Router::new()
        .route("/ready", get(api::ready))
        .route("/fraud-score", post(api::fraud_score))
        .with_state(state);

    axum::serve(listener, app).await.expect("server error");
}
