/// Run decode: process a single token with the existing KV cache.
/// 
/// Returns the logits for the next token prediction.
fn decode(
    model: &Transformer,
    token: u32,
    pos: usize,  // current position in the sequence (for RoPE indexing)
    kv_cache: &mut Vec<KvCache>,
) -> Result<Tensor> {
    // model.forward expects [batch, seq_len] token IDs
    let token_tensor = candle_core::Tensor::new(&[token], &model.device)?
        .reshape((1, 1))?;

    // Forward pass with KV cache (decode phase)
    model.forward(&token_tensor, pos, Some(kv_cache))
}
