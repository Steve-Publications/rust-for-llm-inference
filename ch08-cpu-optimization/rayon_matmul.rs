use rayon::prelude::*;

/// Parallel matrix multiplication using Rayon work-stealing.
pub fn parallel_matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    
    // Split rows across threads
    c.par_chunks_mut(n)
       .enumerate()
       .for_each(|(i, row)| {
           let a_row = &a[i * k..(i + 1) * k];
           for j in 0..n {
               let mut sum = 0.0f32;
               for l in 0..k {
                   sum += a_row[l] * b[l * n + j];
               }
               row[j] = sum;
           }
       });
    
    c
}
