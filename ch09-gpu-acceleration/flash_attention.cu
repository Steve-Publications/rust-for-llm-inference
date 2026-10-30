// Minimal FlashAttention-2 structure (head_dim = 32, block_k <= 64).
// One thread per query position; K/V tiles are staged in SRAM and the
// softmax statistics (m, l) are updated online, so the full score matrix
// is never materialized in HBM. Causal mask: position i attends to 0..i.

extern "C" __global__ void flash_attn_kernel(
    const float* __restrict__ Q,   // [rows, seq_q, head_dim]
    const float* __restrict__ K,   // [rows, seq_k, head_dim]
    const float* __restrict__ V,   // [rows, seq_k, head_dim]
    float* __restrict__ O,         // [rows, seq_q, head_dim]
    int seq_q, int seq_k, int head_dim,
    float scale, int block_k
) {
    const int row = blockIdx.y;   // one block per (batch*head) row
    const int qi  = blockIdx.x;   // one thread per query position

    // Shared memory for tiling: K_tile/V_tile each [block_k][32]
    __shared__ float K_tile[64][32];
    __shared__ float V_tile[64][32];

    // Load the query row into registers
    float qv[32];
    #pragma unroll
    for (int d = 0; d < 32; d++) {
        qv[d] = Q[((size_t)row * seq_q + qi) * head_dim + d];
    }

    // Per-thread online-softmax accumulators
    float m_local = -INFINITY;
    float l_local = 0.0f;
    float o_local[32] = {0.0f};

    // Forward pass: iterate over K/V blocks in SRAM
    for (int j0 = 0; j0 < seq_k; j0 += block_k) {
        const int tk = min(block_k, seq_k - j0);

        // Stage the K and V tiles into SRAM
        for (int i = threadIdx.x; i < tk * head_dim; i += blockDim.x) {
            const int j = i / head_dim, d = i % head_dim;
            K_tile[j][d] = K[((size_t)row * seq_k + j0 + j) * head_dim + d];
            V_tile[j][d] = V[((size_t)row * seq_k + j0 + j) * head_dim + d];
        }
        __syncthreads();

        // Attention scores for this tile: S[j] = scale * q . K_tile[j]
        float S[64];
        for (int j = 0; j < tk; j++) {
            float acc = 0.0f;
            #pragma unroll
            for (int d = 0; d < 32; d++) acc += qv[d] * K_tile[j][d];
            S[j] = acc * scale;
            // Causal mask: future keys are not allowed
            if (j0 + j > qi) S[j] = -INFINITY;
        }

        // Online softmax statistics (rescaling the running max/sum)
        float m_new = m_local;
        for (int j = 0; j < tk; j++) m_new = fmaxf(m_new, S[j]);
        const float corr = expf(m_local - m_new);
        float l_new = l_local * corr;
        for (int j = 0; j < tk; j++) l_new += expf(S[j] - m_new);

        // Accumulate the output, rescaling by corr
        #pragma unroll
        for (int d = 0; d < 32; d++) o_local[d] *= corr;
        for (int j = 0; j < tk; j++) {
            const float p = expf(S[j] - m_new);
            #pragma unroll
            for (int d = 0; d < 32; d++) o_local[d] += p * V_tile[j][d];
        }

        m_local = m_new;
        l_local = l_new;
        __syncthreads();  // done with this tile before the next overwrites it
    }

    // Write the normalized output to global memory
    #pragma unroll
    for (int d = 0; d < 32; d++) {
        O[((size_t)row * seq_q + qi) * head_dim + d] = o_local[d] / l_local;
    }
}
