// Fused RMSNorm + GELU kernel.
// One thread per element. Each thread redundantly computes its row's sum
// of squares (simple and correct; a production kernel would use a
// block-level parallel reduction instead of the per-thread loop).
extern "C" __global__ void rmsnorm_gelu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    float* __restrict__ output,
    int N, int D, float eps
) {
    // Each thread handles one element
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * D) return;
    
    const int i = idx / D;   // row
    const int d = idx % D;   // column
    
    // Compute the sum of squares over the whole row
    float sum_sq = 0.0f;
    for (int t = 0; t < D; t++) {
        float x = input[i * D + t];
        sum_sq += x * x;
    }
    float rms = rsqrtf(sum_sq / D + eps);
    
    // Apply RMSNorm (with gamma) and GELU (tanh approximation) in a single pass
    float normalized = input[idx] * rms * gamma[d];
    float gelu = normalized * (1.0f + tanhf(0.797885f * (normalized + 0.044715f * normalized * normalized * normalized)));
    output[idx] = gelu;
}
