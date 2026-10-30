# Run full inference benchmarks with Criterion.rs
cargo bench --bench bench_inference

# Profile with perf
RUSTFLAGS="-C debuginfo=2" cargo build --release
perf record -g -F 997 -o perf.data cargo bench --bench bench_inference
# Flamegraphs need the perf-script pipeline, not perf report:
perf script -i perf.data | stackcollapse-perf.pl | flamegraph.pl > flamegraph.svg
