use candle_core::{Tensor, Result};
use candle_nn::ops::silu;

/// SwiGLU (Sigmoid-Gated Linear Unit) feed-forward layer.
///
/// FFN(x) = W2 * ( SiLU(W1 * x) ⊙ (W3 * x) )   (⊙ = element-wise)
/// where SiLU(x) = x / (1 + exp(-x))
///
/// This gating mechanism has been shown to improve training stability
/// and final model quality compared to simple GeLU or ReLU.
pub fn swiglu_ffn(
    x: &Tensor,
    w1: &Tensor,   // [ffn_hidden_dim, d_model]
    w3: &Tensor,   // [ffn_hidden_dim, d_model]
    w2: &Tensor,   // [d_model, ffn_hidden_dim]
) -> Result<Tensor> {
    // x: [batch, seq_len, d_model]
    // Candle's matmul requires equal-rank operands, so flatten the batch
    // dims into 2-D matmuls and reshape back.
    let (b, s, i) = x.dims3()?;
    let h = w1.dim(0)?;
    let x2 = x.reshape((b * s, i))?;

    // Project to expanded dimension
    let gate = x2.matmul(&w1.t()?)?.reshape((b, s, h))?;
    let up = x2.matmul(&w3.t()?)?.reshape((b, s, h))?;

    // Apply SiLU (Swish) activation to gate
    let activated = silu(&gate)?;  // SiLU(x) = x / (1 + exp(-x))

    // Element-wise multiply (gate * up)
    let gated = activated.broadcast_mul(&up)?;

    // Project back to d_model
    gated.reshape((b * s, h))?.matmul(&w2.t()?)?.reshape((b, s, i))
}
