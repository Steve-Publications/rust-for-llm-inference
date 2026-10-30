/// Dequantize-and-multiply in a single pass.
///
/// This avoids allocating a separate output buffer by multiplying
/// each dequantized weight directly with the activation and accumulating
/// into the result vector.
///
/// [PERF] In production, this would be implemented as a SIMD kernel
/// that processes 16 bytes (32 INT4 weights) per iteration.
pub fn dequantize_and_multiply(
    input: &[u8],
    scale: f32,
    activations: &[f32],
) -> f32 {
    let mut sum = 0.0f32;
    let mut idx = 0usize;

    for &byte in input {
        // Process 2 INT4 weights per byte; idx is the GLOBAL weight index
        // (activations[i] would reuse indices 0..2 for every byte).
        for i in 0..2 {
            let nib = ((byte >> (i * 4)) & 0xF) as i8;
            let val = nib - 8;
            sum += (val as f32) * scale * activations[idx];
            idx += 1;
        }
    }

    sum
}
