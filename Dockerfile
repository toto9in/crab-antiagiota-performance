# syntax=docker/dockerfile:1.7

FROM lukemathwalker/cargo-chef:latest-rust-1 AS chef
WORKDIR /app

FROM chef AS planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS builder
COPY --from=planner /app/recipe.json recipe.json
ENV RUSTFLAGS="-C target-cpu=haswell"
RUN cargo chef cook --release --recipe-path recipe.json
COPY . .
RUN cargo build --release --bin crab-antiagiota-performance --bin build-index
# Build the IVF index once, at image-build time, so the API never pays the
# k-means cost on boot — it just loads index.bin.
RUN ./target/release/build-index /app/index.bin

FROM debian:trixie-slim AS runtime
WORKDIR /app

# Cap glibc malloc arenas. The 3M-row parse churns millions of transient
# allocations; with default arenas (8 × nproc) the freed pages are retained as
# fragmentation and RSS balloons to ~550 MB. Arena=1 holds peak RSS to ~150 MB.
ENV MALLOC_ARENA_MAX=1

COPY --from=builder /app/target/release/crab-antiagiota-performance /usr/local/bin/crab-antiagiota-performance
COPY --from=builder /app/index.bin /app/index.bin

ENV INDEX_PATH=/app/index.bin

ENTRYPOINT ["/usr/local/bin/crab-antiagiota-performance"]
