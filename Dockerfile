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
RUN cargo build --release --bin crab-antiagiota-performance

FROM debian:trixie-slim AS runtime
WORKDIR /app

COPY --from=builder /app/target/release/crab-antiagiota-performance /usr/local/bin/crab-antiagiota-performance

ENTRYPOINT ["/usr/local/bin/crab-antiagiota-performance"]
