// kernels/dequant_matmul.cu
// 
// Dequantize INT2 weights and multiply with FP16 activations in one pass.
//
// A: [M x K] INT2 (packed: 2 bits per weight, 4 weights per byte)
// B: [K x N] FP16
// C: [M x N] FP32 output (accumulation in FP32 for numerical stability)
// 
// scales: per-group scale factors [M / block_size x N] in FP16
// block_size: number of weights per scale factor (typically 128)
// 
// Launch config:
//   Block: (16, 16) -- one thread per output element
//   Grid:  ((N+15)/16, (M+15)/16)

#include <cstdint>
#include <cuda_fp16.h>

extern "C" __global__ void dequant_matmul_kernel(
    const uint8_t* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    const half* __restrict__ scales,
    int M, int N, int K, int block_size
) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;  // output column
    const int row = blockIdx.y * blockDim.y + threadIdx.y;  // output row
    
    if (row >= M || col >= N) return;
    
    // Determine which scale factor applies to this output row
    const int scale_idx = row / block_size;
    const float scale = __half2float(scales[scale_idx * N + col]);
    
    float sum = 0.0f;
    const int K_packed = (K + 3) / 4;  // bytes needed for K weights
    
    // Process 4 weights per byte, unrolled for better instruction-level parallelism
    const int full_bytes = K / 4;
    for (int packed_col = 0; packed_col < full_bytes; packed_col++) {
        const uint8_t byte = A[row * K_packed + packed_col];
        
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            // Extract 2 bits; with offset -2 the range is [-2, 1]
            const int val = ((byte >> (i * 2)) & 0x3) - 2;
            
            // Load B activation and multiply-accumulate
            const int b_idx = (packed_col * 4 + i) * N + col;
            sum += (float)val * scale * __half2float(B[b_idx]);
        }
    }
    
    // Handle remaining weights (< 4)
    for (int packed_col = full_bytes; packed_col < K_packed; packed_col++) {
        uint8_t byte = A[row * K_packed + packed_col];
        const int base_idx = packed_col * 4;
        
        #pragma unroll
        for (int i = 0; i < 4 && (base_idx + i) < K; i++) {
            int val = ((byte >> (i * 2)) & 0x3) - 2;
            const int b_idx = (base_idx + i) * N + col;
            sum += (float)val * scale * __half2float(B[b_idx]);
        }
    }
    
    C[row * N + col] = sum;
}
