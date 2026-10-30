use candle_core::{Tensor, Result};
use candle_nn::ops::softmax_last_dim;

/// Compute scaled dot-product attention.
/// 
/// Inputs:
///   q: [batch, heads, seq_len_q, head_dim]
///   k: [batch, heads, seq_len_k, head_dim]
///   v: [batch, heads, seq_len_k, head_dim]
/// Output:
///   attention output: [batch, heads, seq_len_q, head_dim]
pub fn scaled_dot_product_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
) -> Result<Tensor> {
    let d_k = q.dim(3)?;
    let scale = 1.0 / (d_k as f64).sqrt();
    
    // Q @ K^T: [batch, heads, seq_len_q, seq_len_k]
    let attn_scores = q.matmul(&k.transpose(2, 3)?)?;
    let attn_scores = attn_scores.affine(scale, 0.0)?;
    
    // Apply softmax along the last dimension (seq_len_k)
    let attn_weights = softmax_last_dim(&attn_scores)?;
    
    // attn_weights @ V: [batch, heads, seq_len_q, head_dim]
    attn_weights.matmul(v)
}
