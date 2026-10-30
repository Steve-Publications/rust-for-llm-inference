#![feature(portable_simd)]
use std::simd::{f32x4, prelude::SimdFloat};

/// Dot product of two slices, four lanes at a time.
fn simd_dot_product(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = f32x4::splat(0.0);
    // Only the shared prefix can be multiplied: with slices of different
    // length the vector loop and the tail would run off the end of the
    // shorter one.
    let len = a.len().min(b.len());
    // Vectorize over the complete 4-element chunks...
    let vec_len = len - (len % 4);
    for chunk in a[..vec_len].chunks_exact(4).zip(b[..vec_len].chunks_exact(4)) {
        let va = f32x4::from_slice(chunk.0);
        let vb = f32x4::from_slice(chunk.1);
        sum += va * vb;
    }
    // ...and handle the remainder scalar (chunks_exact would drop it).
    let tail: f32 = a[vec_len..len].iter().zip(&b[vec_len..len]).map(|(x, y)| x * y).sum();
    sum.reduce_sum() + tail
}
