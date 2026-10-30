use candle_core::{Tensor, Result, D};

/// Root Mean Square Layer Normalization (RMSNorm).
///
/// RMSNorm(x) = x / sqrt(mean(x^2) + eps) * gamma
///
/// This is a simplified form of LayerNorm that omits mean subtraction
/// and bias addition, reducing compute by ~30%.
pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    // x: [batch, seq_len, d_model]
    // Compute mean of squares along the last dimension
    let v = x.sqr()?.mean_keepdim(D::Minus1)?;
    // eps as a scalar tensor in x's dtype so the addition type-checks
    let eps_t = Tensor::full(eps, (), x.device())?.to_dtype(x.dtype())?;
    // Normalize: x / sqrt(v + eps); powf(-0.5) is 1/sqrt in one step
    let inv_rms = v.broadcast_add(&eps_t)?.powf(-0.5)?;
    let x_normed = x.broadcast_mul(&inv_rms)?;
    // Scale by learnable gamma
    x_normed.broadcast_mul(weight)
}
