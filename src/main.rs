use std::{
    env, fs,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    sync::Arc,
};

use crab_antiagiota_p99::{
    api, classifier::FraudDetector, dataset::ReferenceDataset, distance::DistanceEngine,
    state::AppState,
};
use tokio::{net::UnixListener, runtime::Builder, signal};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let runtime_config = RuntimeConfig::from_env()?;
    let runtime = runtime_config.build()?;

    runtime.block_on(async_main(runtime_config))
}

async fn async_main(runtime_config: RuntimeConfig) -> Result<(), Box<dyn std::error::Error>> {
    let socket_path = env::var("SOCKET_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp/sockets/api.sock"));

    prepare_socket_path(&socket_path)?;

    let dataset = Arc::new(ReferenceDataset::load_embedded()?);
    let distance = DistanceEngine::from_env();
    let detector = Arc::new(FraudDetector::new(dataset.clone(), distance));
    let app = api::router(AppState::new(detector));

    if runtime_config.startup_diagnostics {
        eprintln!(
            "startup: runtime={} worker_threads={} distance={} dataset_records={}",
            runtime_config.flavor.as_str(),
            runtime_config.worker_threads.unwrap_or(1),
            distance.name(),
            dataset.len()
        );
    }

    let listener = UnixListener::bind(&socket_path)?;
    fs::set_permissions(&socket_path, fs::Permissions::from_mode(0o777))?;
    let socket_guard = SocketGuard::new(socket_path.clone());

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    drop(socket_guard);

    Ok(())
}

#[derive(Clone, Copy)]
struct RuntimeConfig {
    flavor: RuntimeFlavor,
    worker_threads: Option<usize>,
    startup_diagnostics: bool,
}

impl RuntimeConfig {
    fn from_env() -> Result<Self, Box<dyn std::error::Error>> {
        let flavor = match env::var("TOKIO_RUNTIME").ok().as_deref() {
            Some("current_thread") => RuntimeFlavor::CurrentThread,
            Some("multi_thread") | None => RuntimeFlavor::MultiThread,
            Some(other) => {
                return Err(format!(
                    "invalid TOKIO_RUNTIME={other:?}; expected current_thread or multi_thread"
                )
                .into());
            }
        };

        let worker_threads = match flavor {
            RuntimeFlavor::CurrentThread => None,
            RuntimeFlavor::MultiThread => Some(
                env::var("TOKIO_WORKER_THREADS")
                    .ok()
                    .map(|raw| raw.parse::<usize>())
                    .transpose()?
                    .filter(|threads| *threads > 0)
                    .unwrap_or_else(default_worker_threads),
            ),
        };

        Ok(Self {
            flavor,
            worker_threads,
            startup_diagnostics: env_flag("STARTUP_DIAGNOSTICS"),
        })
    }

    fn build(self) -> Result<tokio::runtime::Runtime, std::io::Error> {
        let mut builder = match self.flavor {
            RuntimeFlavor::CurrentThread => Builder::new_current_thread(),
            RuntimeFlavor::MultiThread => {
                let mut builder = Builder::new_multi_thread();
                builder.worker_threads(self.worker_threads.unwrap_or(1));
                builder
            }
        };
        builder.enable_all();
        builder.build()
    }
}

#[derive(Clone, Copy)]
enum RuntimeFlavor {
    CurrentThread,
    MultiThread,
}

impl RuntimeFlavor {
    fn as_str(self) -> &'static str {
        match self {
            Self::CurrentThread => "current_thread",
            Self::MultiThread => "multi_thread",
        }
    }
}

fn default_worker_threads() -> usize {
    std::thread::available_parallelism()
        .map(usize::from)
        .map(|threads| threads.min(2))
        .unwrap_or(1)
}

fn env_flag(name: &str) -> bool {
    matches!(
        env::var(name).as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
    )
}

fn prepare_socket_path(socket_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = socket_path.parent() {
        fs::create_dir_all(parent)?;
    }

    match fs::remove_file(socket_path) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(Box::new(error)),
    }

    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = signal::ctrl_c().await;
    };

    #[cfg(unix)]
    let terminate = async {
        if let Ok(mut sigterm) = signal::unix::signal(signal::unix::SignalKind::terminate()) {
            let _ = sigterm.recv().await;
        }
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {}
        _ = terminate => {}
    }
}

struct SocketGuard {
    path: PathBuf,
}

impl SocketGuard {
    fn new(path: PathBuf) -> Self {
        Self { path }
    }
}

impl Drop for SocketGuard {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}
