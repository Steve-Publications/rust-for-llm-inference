/// Run prefill: process the full prompt in one pass.
/// 
/// Returns the logits for every prompt position (the last one predicts the
/// first output token) and updates the KV cache with all prompt tokens'
/// K and V vectors.
fn prefill(
    model: &Transformer,
    tokens: &[u32],
    kv_cache: &mut Vec<KvCache>,
) -> Result<Tensor> {
    // model.forward expects [batch, seq_len] token IDs
    let token_tensor = candle_core::Tensor::new(tokens, &model.device)?
        .reshape((1, tokens.len()))?;

    // Forward pass with the KV cache (prefill phase)
    // The KV cache is populated internally by the attention layers
    // (they append their K and V vectors for each token in the prompt)
    model.forward(&token_tensor, 0, Some(kv_cache))
}
