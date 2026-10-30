use candle_core::{Tensor, Result};

/// Fully connected (linear) layer: y = x @ W^T + b
///
/// Memory layout note: Candle stores weights in [out_features, in_features]
/// format, so we transpose during matmul to get the correct shape.
pub fn linear(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    // x: [batch, seq_len, in_features]
    // weight: [out_features, in_features]
    // Result: [batch, seq_len, out_features]
    //
    // Candle's matmul requires both operands to have the same number of
    // dimensions, so we flatten the batch dims into a plain 2-D matmul and
    // reshape the result back.
    let (b, s, i) = x.dims3()?;
    let o = weight.dim(0)?;
    let out = x
        .reshape((b * s, i))?
        .matmul(&weight.t()?)?
        .reshape((b, s, o))?;

    match bias {
        Some(bias) => out.broadcast_add(bias),
        None => Ok(out),
    }
}
