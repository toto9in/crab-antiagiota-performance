# crab-antiagiota-p99

Implementacao Rust p99-first da rinha com `nginx` + `api1` + `api2`, sem banco, com dataset de 100k referencias embutido no binario e busca exata `k=5` totalmente in-process.

## Rodando localmente

```bash
cargo test
cargo run
```

Para subir a topologia completa:

```bash
docker compose up --build
```

## Runtime e diagnostico

O binario aceita alguns ajustes de execucao para benchmark local:

```bash
DISTANCE_IMPL=avx2 cargo run
TOKIO_RUNTIME=multi_thread TOKIO_WORKER_THREADS=2 cargo run
STARTUP_DIAGNOSTICS=1 cargo run
```

- `DISTANCE_IMPL`: `auto` (padrao), `avx2` ou `scalar`
- `TOKIO_RUNTIME`: `multi_thread` (padrao) ou `current_thread`
- `TOKIO_WORKER_THREADS`: quantidade de workers do runtime multithread
- `STARTUP_DIAGNOSTICS=1`: imprime runtime, engine de distancia e tamanho do dataset na inicializacao
