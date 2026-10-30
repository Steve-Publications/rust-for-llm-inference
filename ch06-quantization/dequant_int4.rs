/// Dequantize INT4 weights to FP32.
///
/// [PERF] This is the naive implementation. In production, this should be
/// replaced with a SIMD-optimized kernel (see Chapter 8 for CPU optimization
/// and Chapter 9 for CUDA kernels).
///
/// input: packed INT4 values (2 per byte, 4 bits each)
/// scale: per-group scale factor
/// group_size: number of weights per scale factor (the single `scale`
/// here applies to the whole input; in a grouped scheme, one scale
/// is applied per group and this parameter selects the group stride)
pub fn dequantize_int4_naive(input: &[u8], scale: f32, group_size: usize) -> Vec<f32> {
    let _ = group_size; // single shared scale for the whole input (naive case)
    let mut output = Vec::with_capacity(input.len() * 2);

    for &byte in input {
        // Extract 2 nibbles (4 bits each) from a single byte
        for i in 0..2 {
            // Extract 4 bits, bias to the signed range [-8, 7]
            let nib = ((byte >> (i * 4)) & 0xF) as i8;
            let val = nib - 8;
            output.push((val as f32) * scale);
        }
    }

    output
}
