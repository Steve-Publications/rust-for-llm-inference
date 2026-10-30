use candle_core::{Result, Tensor};

/// Transform weight layout from training format to inference format.
///
/// Training format: [d_model, d_head * n_heads] (row-major)
/// Inference format: [n_heads, d_head, d_model] (for better cache locality
/// when iterating over heads during attention computation)
pub fn reshape_attention_weights(
    weights: &Tensor,
    num_heads: usize,
    head_dim: usize,
    d_model: usize,
) -> Result<Tensor> {
    // Original shape: [d_model, d_head * n_heads] = [4096, 4096]
    // Reshape to: [d_model, n_heads, d_head] = [4096, 32, 128]
    let reshaped = weights.reshape((d_model, num_heads, head_dim))?;

    // Transpose to: [n_heads, d_model, d_head] = [32, 4096, 128]
    let transposed = reshaped.permute([1, 0, 2])?;

    // Move the d_model axis to the end: [n_heads, d_head, d_model] = [32, 128, 4096].
    // Note: this must be a permute, not a reshape - reshaping a permuted
    // view would reinterpret the memory layout and silently give wrong
    // data. The result is non-contiguous; materialize it with contiguous()
    // before it is used in matmuls.
    transposed.permute([0, 2, 1])?.contiguous()
}
