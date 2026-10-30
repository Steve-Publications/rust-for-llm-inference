#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// AVX2-optimized matrix multiplication kernel.
/// 
/// Processes 8 f32 values per instruction using ymm registers,
/// 4 blocks of 8 columns at a time (32 columns per outer iteration).
/// Requires: AVX2 (256-bit SIMD), FMA3.
///
/// # Safety
/// The caller must confirm at runtime that the CPU supports both features,
/// which is exactly what a safe wrapper around this kernel does:
///
/// ```ignore
/// if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
///     unsafe { avx2_matmul_f32(a, b, m, n, k) }
/// } else {
///     simd_matmul_f32(a, b, m, n, k)   // portable fallback (ch08-02)
/// }
/// ```
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn avx2_matmul_f32(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    
    unsafe {
        for i in 0..m {
            let a_row = &a[i * k..(i + 1) * k];
            let c_row_start = i * n;
            
            // Process 32 columns at a time (4 ymm accumulators x 8 f32 each)
            for j_base in (0..n).step_by(32) {
                if j_base + 32 > n { break; }
                
                let mut sum0 = _mm256_setzero_ps();
                let mut sum1 = _mm256_setzero_ps();
                let mut sum2 = _mm256_setzero_ps();
                let mut sum3 = _mm256_setzero_ps();
                
                for l in 0..k {
                    let a_val = _mm256_set1_ps(a_row[l]);
                    let b_col0 = _mm256_loadu_ps(&b[l * n + j_base]);
                    let b_col1 = _mm256_loadu_ps(&b[l * n + j_base + 8]);
                    let b_col2 = _mm256_loadu_ps(&b[l * n + j_base + 16]);
                    let b_col3 = _mm256_loadu_ps(&b[l * n + j_base + 24]);
                    
                    sum0 = _mm256_fmadd_ps(a_val, b_col0, sum0);
                    sum1 = _mm256_fmadd_ps(a_val, b_col1, sum1);
                    sum2 = _mm256_fmadd_ps(a_val, b_col2, sum2);
                    sum3 = _mm256_fmadd_ps(a_val, b_col3, sum3);
                }
                
                // Store the four 8-column results
                _mm256_storeu_ps(&mut c[c_row_start + j_base], sum0);
                _mm256_storeu_ps(&mut c[c_row_start + j_base + 8], sum1);
                _mm256_storeu_ps(&mut c[c_row_start + j_base + 16], sum2);
                _mm256_storeu_ps(&mut c[c_row_start + j_base + 24], sum3);
            }
            
            // Scalar fallback for remaining columns
            for j in (n / 32 * 32)..n {
                let mut sum = 0.0f32;
                for l in 0..k {
                    sum += a_row[l] * b[l * n + j];
                }
                c[c_row_start + j] = sum;
            }
        }
    }
    
    c
}
