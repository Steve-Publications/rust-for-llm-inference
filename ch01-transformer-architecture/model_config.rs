/// Hyperparameters for a decoder-only transformer model.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Vocabulary size (number of tokens in the tokenizer).
    pub vocab_size: usize,
    /// Number of layers (depth of the network).
    pub num_layers: usize,
    /// Hidden dimension (d_model).
    pub hidden_size: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Number of KV heads (for GQA; equals num_heads for MHA).
    pub num_kv_heads: usize,
    /// Dimension of each attention head (d_k = d_v).
    pub head_dim: usize,
    /// Feed-forward expansion factor. The intermediate dimension
    /// is (hidden_size * ffn_hidden_dim / hidden_size) for SwiGLU.
    pub ffn_hidden_dim: usize,
    /// Maximum context length the model was trained on.
    pub max_context_length: usize,
    /// Number of attention heads to use for RoPE.
    pub rope_head_count: usize,
    /// Base frequency for RoPE (typically 1_000_000).
    pub rope_theta: f64,
    /// Epsilon for RMSNorm.
    pub rms_norm_eps: f64,
}

impl ModelConfig {
    /// Create a configuration matching Llama 3 8B.
    pub fn llama3_8b() -> Self {
        Self {
            vocab_size: 128_256,
            num_layers: 32,
            hidden_size: 4_096,
            num_heads: 32,
            num_kv_heads: 8, // GQA with factor 4
            head_dim: 128,
            ffn_hidden_dim: 14_336,
            max_context_length: 8_192,
            rope_head_count: 32,
            rope_theta: 500_000.0,
            rms_norm_eps: 1e-5,
        }
    }

    /// Create a configuration with GPT-2 small's dimensions (124M
    /// parameters). Used in the benchmarking section for a small
    /// end-to-end run on CPU.
    pub fn gpt2_small() -> Self {
        Self {
            vocab_size: 50_257,
            num_layers: 12,
            hidden_size: 768,
            num_heads: 12,
            num_kv_heads: 12, // MHA (GPT-2 has no GQA)
            head_dim: 64,
            ffn_hidden_dim: 3_072,
            max_context_length: 1_024,
            rope_head_count: 12,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-5,
        }
    }

    /// Total number of parameters in the model (excluding KV cache).
    pub fn param_count(&self) -> usize {
        // Embedding + output projection
        let embedding = self.vocab_size * self.hidden_size;
        let output_proj = self.vocab_size * self.hidden_size;

        // Per layer: attention projections + FFN + norms (norms are small, omitted for estimate)
        let per_layer = {
            // WQ: hidden x (num_heads * head_dim)
            let wq = self.hidden_size * self.num_heads * self.head_dim;
            // WK: hidden x (num_kv_heads * head_dim)
            let wk = self.hidden_size * self.num_kv_heads * self.head_dim;
            // WV: hidden x (num_kv_heads * head_dim)
            let wv = self.hidden_size * self.num_kv_heads * self.head_dim;
            // WO: (num_heads * head_dim) x hidden
            let wo = self.num_heads * self.head_dim * self.hidden_size;
            // W1, W3: hidden x ffn_hidden_dim each
            let w1 = self.hidden_size * self.ffn_hidden_dim;
            let w3 = self.hidden_size * self.ffn_hidden_dim;
            // W2: ffn_hidden_dim x hidden
            let w2 = self.ffn_hidden_dim * self.hidden_size;
            wq + wk + wv + wo + w1 + w3 + w2
        };

        embedding + output_proj + self.num_layers * per_layer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llama3_8b_param_count() {
        let config = ModelConfig::llama3_8b();
        // ~8.03B parameters for Llama 3 8B
        assert!(config.param_count() > 7_500_000_000);
        assert!(config.param_count() < 8_500_000_000);
    }
}
