use std::env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = candle_core::Device::new_cuda(0)?;
    let config = ModelConfig::llama3_8b();

    // Fail fast if the model cannot be loaded.
    let transformer = Transformer::load(&config, &env::var("MODEL_PATH")?, &device)
        .map_err(|e| format!("Failed to load model: {}", e))?;

    let tokenizer = LlmTokenizer::from_file("tokenizer.model")
        .map_err(|e| format!("Failed to load tokenizer: {}", e))?;

    // ... continue with server setup
    Ok(())
}
