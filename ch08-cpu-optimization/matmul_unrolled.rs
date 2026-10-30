/// Matrix multiplication C = A × B with 4-way loop unrolling.
///
/// A: [M x K], B: [K x N], C: [M x N] (row-major layout)
///
/// This is a portable scalar kernel: the four partial sums stay in
/// registers (register blocking) and on modern targets the compiler
/// vectorizes the inner loop automatically. For guaranteed SIMD, use
/// the AVX2 kernel below (x86_64) or a NEON kernel (AArch64).
pub fn simd_matmul_f32(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    
    // Process 4 columns at a time (AVX2 processes 8 f32s, but we use 4
    // for better register utilization with the inner loop)
    for i in 0..m {
        let a_row = &a[i * k..(i + 1) * k];
        let c_row_start = i * n;
        
        for j_base in (0..n).step_by(4) {
            if j_base + 4 > n { break; }
            
            let mut sum0 = 0.0f32;
            let mut sum1 = 0.0f32;
            let mut sum2 = 0.0f32;
            let mut sum3 = 0.0f32;
            
            for l in 0..k {
                let a_val = a_row[l];
                sum0 += a_val * b[l * n + j_base];
                sum1 += a_val * b[l * n + j_base + 1];
                sum2 += a_val * b[l * n + j_base + 2];
                sum3 += a_val * b[l * n + j_base + 3];
            }
            
            c[c_row_start + j_base] = sum0;
            c[c_row_start + j_base + 1] = sum1;
            c[c_row_start + j_base + 2] = sum2;
            c[c_row_start + j_base + 3] = sum3;
        }
        
        // Handle remaining columns (scalar fallback)
        for j in (n / 4 * 4)..n {
            let mut sum = 0.0f32;
            for l in 0..k {
                sum += a_row[l] * b[l * n + j];
            }
            c[c_row_start + j] = sum;
        }
    }
    
    c
}
