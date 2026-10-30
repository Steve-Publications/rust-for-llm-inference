/// Owns the model, the tokenizer and the device. The HTTP handlers
/// below call its `generate` / `generate_stream` methods.
pub struct InferenceEngine {
    pub transformer: Transformer,
    pub tokenizer: LlmTokenizer,
    pub device: candle_core::Device,
}

/// Full result of a completed generation.
#[derive(Debug, Clone)]
pub struct GenerationResult {
    pub text: String,
    pub tokens: Vec<u32>,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
}

/// One chunk of a streamed response.
#[derive(serde::Serialize, Debug)]
pub struct StreamChunk {
    pub text: String,
    pub token_id: u32,
    pub is_final: bool,
}

impl InferenceEngine {
    pub fn new(
        transformer: Transformer,
        tokenizer: LlmTokenizer,
        device: candle_core::Device,
    ) -> Self {
        Self { transformer, tokenizer, device }
    }

    /// Run the prefill + decode loop from Chapter 4 and return the
    /// complete result in one call.
    pub fn generate(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f64,
        top_p: f64,
    ) -> Result<GenerationResult, candle_core::Error> {
        let prompt_tokens = self.tokenizer.encode(prompt, true);

        // Fresh per-layer KV caches (same setup as Chapter 4's `generate`)
        let empty = candle_core::Tensor::from_vec(
            Vec::<f32>::new(),
            (1, 0, self.transformer.config.num_kv_heads, self.transformer.config.head_dim),
            &self.device,
        )?;
        let mut kv_cache: Vec<KvCache> = (0..self.transformer.config.num_layers)
            .map(|_| KvCache::new(empty.clone(), empty.clone()))
            .collect();

        // Prefill the whole prompt at once
        let token_tensor = candle_core::Tensor::new(&prompt_tokens[..], &self.device)?
            .reshape((1, prompt_tokens.len()))?;
        let mut logits = self
            .transformer
            .forward(&token_tensor, 0, Some(&mut kv_cache))?;

        // Decode until EOS or the token limit is reached
        let mut generated_tokens = Vec::with_capacity(max_tokens);
        let mut pos = prompt_tokens.len();
        for step in 0..max_tokens {
            let next_token = Transformer::sample_logits(&logits, temperature, top_p)?;
            generated_tokens.push(next_token);
            if next_token == self.tokenizer.eos_token_id || step + 1 == max_tokens {
                break;
            }
            let token_tensor = candle_core::Tensor::new(&[next_token], &self.device)?
                .reshape((1, 1))?;
            logits = self
                .transformer
                .forward(&token_tensor, pos, Some(&mut kv_cache))?;
            pos += 1;
        }

        let text = self
            .tokenizer
            .decode(&generated_tokens, true)
            .map_err(candle_core::Error::Msg)?;
        Ok(GenerationResult {
            text,
            tokens: generated_tokens.clone(),
            prompt_tokens: prompt_tokens.len(),
            generated_tokens: generated_tokens.len(),
        })
    }

    /// The same loop, but yields one `StreamChunk` per generated token
    /// instead of waiting for the full sequence.
    pub fn generate_stream(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f64,
        top_p: f64,
    ) -> std::pin::Pin<
        Box<dyn futures_util::Stream<Item = Result<StreamChunk, candle_core::Error>> + Send + '_>,
    > {
        let prompt_tokens = self.tokenizer.encode(prompt, true);
        Box::pin(async_stream::stream! {
            let engine = self;
            let empty = match candle_core::Tensor::from_vec(
                Vec::<f32>::new(),
                (1, 0, engine.transformer.config.num_kv_heads,
                 engine.transformer.config.head_dim),
                &engine.device,
            ) {
                Ok(t) => t,
                Err(e) => { yield Err(e); return; }
            };
            let mut kv_cache: Vec<KvCache> = (0..engine.transformer.config.num_layers)
                .map(|_| KvCache::new(empty.clone(), empty.clone()))
                .collect();
            let token_tensor = match candle_core::Tensor::new(&prompt_tokens[..], &engine.device)
                .and_then(|t| t.reshape((1, prompt_tokens.len())))
            {
                Ok(t) => t,
                Err(e) => { yield Err(e); return; }
            };
            let mut logits = match engine.transformer
                .forward(&token_tensor, 0, Some(&mut kv_cache))
            {
                Ok(l) => l,
                Err(e) => { yield Err(e); return; }
            };

            let mut pos = prompt_tokens.len();
            for step in 0..max_tokens {
                let next_token = match Transformer::sample_logits(&logits, temperature, top_p) {
                    Ok(t) => t,
                    Err(e) => { yield Err(e); break; }
                };
                let is_final = next_token == engine.tokenizer.eos_token_id
                    || step + 1 == max_tokens;
                let text = engine.tokenizer
                    .decode(&[next_token], false)
                    .unwrap_or_default();
                yield Ok(StreamChunk { text, token_id: next_token, is_final });
                if is_final { break; }
                let token_tensor = match candle_core::Tensor::new(&[next_token], &engine.device)
                    .and_then(|t| t.reshape((1, 1)))
                {
                    Ok(t) => t,
                    Err(e) => { yield Err(e); break; }
                };
                logits = match engine.transformer
                    .forward(&token_tensor, pos, Some(&mut kv_cache))
                {
                    Ok(l) => l,
                    Err(e) => { yield Err(e); break; }
                };
                pos += 1;
            }
        })
    }
}
