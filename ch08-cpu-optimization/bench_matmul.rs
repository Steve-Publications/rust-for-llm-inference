use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use llm_rs::{generate, LlmTokenizer, ModelConfig, Transformer};

fn benchmark_full_inference(c: &mut Criterion) {
    let device = candle_core::Device::Cpu;
    
    // Load a small model (GPT-2 small, 124M parameters)
    let config = ModelConfig::gpt2_small();
    let model = Transformer::load(&config, "gpt2-small.safetensors", &device).unwrap();
    let tokenizer = LlmTokenizer::from_file("gpt2-tokenizer.json").unwrap();
    
    let mut group = c.benchmark_group("inference");
    
    // Test with different prompt lengths and output lengths
    for (prompt_len, output_len) in [(64, 32), (256, 128), (1024, 256)] {
        // GPT-2's BPE encodes " The" as one token, so one "The " per
        // token of the target prompt length (repeating "The" with no space
        // merges into longer multi-word tokens and undershoots badly).
        let prompt = "The ".repeat(prompt_len); // ~prompt_len tokens
        group.bench_with_input(
            BenchmarkId::new(format!("pp{}_tg{}", prompt_len, output_len), "cpu"),
            &(prompt, output_len),
            |b, (prompt_text, output_len)| {
                // The input is &(String, usize); match-ergonomics binds
                // prompt_text: &String and output_len: &usize.
                b.iter(|| {
                    let _ = generate(&model, &tokenizer, prompt_text, *output_len, 0.7, 0.9).unwrap();
                });
            },
        );
    }
    
    group.finish();
}

criterion_group!(benches, benchmark_full_inference);
criterion_main!(benches);
