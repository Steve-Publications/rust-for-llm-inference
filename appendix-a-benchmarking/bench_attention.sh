# Run with Criterion.rs, 100 iterations per benchmark
cargo bench --bench bench_attention

# Verify O(n²) scaling by plotting sequence length vs. time
python3 plot_scaling.py --input criterion-output/attention.json
