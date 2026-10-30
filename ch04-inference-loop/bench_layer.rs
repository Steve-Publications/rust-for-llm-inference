use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};

fn benchmark_attention(c: &mut Criterion) {
    // Metal on Apple GPUs; falls back to CPU elsewhere
    let device = candle_core::Device::metal_if_available(0).unwrap_or_else(|_| candle_core::Device::Cpu);
    
    // Create test tensors: batch=1, heads=32, seq_len varies, head_dim=128
    let head_dim = 128;
    let num_heads = 32;
    
    let mut group = c.benchmark_group("attention");
    
    for seq_len in [64, 128, 256, 512, 1024, 2048] {
        let q = candle_core::Tensor::randn(0.0, 1.0, (1, num_heads, seq_len, head_dim), &device).unwrap();
        let k = candle_core::Tensor::randn(0.0, 1.0, (1, num_heads, seq_len, head_dim), &device).unwrap();
        let v = candle_core::Tensor::randn(0.0, 1.0, (1, num_heads, seq_len, head_dim), &device).unwrap();
        
        group.bench_with_input(BenchmarkId::from_parameter(seq_len), &seq_len, |b, _| {
            b.iter(|| {
                        // Q @ K^T / sqrt(d_k)
                        let qk = q.matmul(&k.transpose(2, 3).unwrap()).unwrap();
                        let scale = 1.0 / (head_dim as f64).sqrt();
                        let qk_scaled = qk.affine(scale, 0.0).unwrap();
                        // softmax
                        let attn = candle_nn::ops::softmax_last_dim(&qk_scaled).unwrap();
                        // attn @ V
                        attn.matmul(&v).unwrap()
                    });
        });
    }
    
    group.finish();
}

criterion_group!(benches, benchmark_attention);
criterion_main!(benches);
