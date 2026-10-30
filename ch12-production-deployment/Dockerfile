# Stage 1: Build with full Rust toolchain
FROM rust:1.99-slim AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config libssl-dev build-essential \
    && rm -rf /var/lib/apt/lists/*

# Cache dependencies (Cargo.toml rarely changes): a dummy build compiles
# the whole dependency tree once, so the real build below only recompiles
# the crate itself when sources change.
COPY Cargo.toml Cargo.lock ./
RUN mkdir src && echo "fn main() {}" > src/main.rs
ARG CARGO_PROFILE=release
RUN cargo build --$CARGO_PROFILE

# Copy source and build (only the crate is recompiled)
COPY . .
RUN touch src/main.rs && cargo build --$CARGO_PROFILE

# Stage 2: Runtime image (minimal)
FROM debian:bookworm-slim AS runtime
WORKDIR /app

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libstdc++6 libgcc-s1 ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the binary and model files
COPY --from=builder /app/target/release/llm-rs-server .
COPY --from=builder /app/models ./models

EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

ENTRYPOINT ["./llm-rs-server"]
