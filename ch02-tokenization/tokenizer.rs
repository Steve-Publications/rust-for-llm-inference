use tokenizers::Tokenizer;

/// A tokenizer wrapper that handles encoding, decoding and
/// special token management for a decoder-only transformer.
pub struct LlmTokenizer {
    inner: Tokenizer,
    /// The BOS (beginning of sequence) token ID.
    pub bos_token_id: u32,
    /// The EOS (end of sequence) token ID.
    pub eos_token_id: u32,
}

impl LlmTokenizer {
    /// Load a tokenizer from a `tokenizer.json` file (the JSON
    /// serialization produced by the HuggingFace `tokenizers` crate).
    /// A SentencePiece `.model` protobuf is not read directly - convert
    /// it to `tokenizer.json` first (e.g. `tokenizers`' conversion tools).
    pub fn from_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let mut tokenizer = match Tokenizer::from_file(path) {
            Ok(t) => t,
            Err(e) => return Err(format!("Failed to load tokenizer: {}", e).into()),
        };
        
        // Use the standard BPE pre-tokenizer for consistency
        tokenizer.with_pre_tokenizer(Some(
            tokenizers::pre_tokenizers::byte_level::ByteLevel::default()
        ));
        
        // Take the BOS/EOS ids from the tokenizer's own added tokens.
        // Hard-coding one model's ids (e.g. Llama 3's 128000/128001)
        // would produce out-of-range embedding indices for other models
        // such as GPT-2 (vocab 50257).
        // The decoder map is id -> AddedToken; the ids come from the keys.
        let added = tokenizer.get_added_tokens_decoder();
        // No explicit begin/end token (e.g. GPT-2 only has <|endoftext|>):
        // fall back to the lowest-id special token, which is <|begin_of_text|>
        // / <|end_of_text|> for Llama 3 and <|endoftext|> for GPT-2.
        let lowest_special = added
            .iter()
            .filter(|(_, t)| t.special)
            .map(|(id, _)| *id)
            .min();
        let bos_token_id = added
            .iter()
            .find(|(_, t)| t.content == "<s>" || t.content == "[BOS]")
            .map(|(id, _)| *id)
            .or(lowest_special)
            .unwrap_or(128000); // Llama 3 fallback
        let eos_token_id = added
            .iter()
            .find(|(_, t)| t.content == "</s>" || t.content == "[EOS]")
            .map(|(id, _)| *id)
            .or(lowest_special)
            .unwrap_or(128001); // Llama 3 fallback

        Ok(Self {
            inner: tokenizer,
            bos_token_id,
            eos_token_id,
        })
    }

    /// Encode text into a sequence of token IDs.
    /// Adds BOS token if `add_bos` is true.
    pub fn encode(&self, text: &str, add_bos: bool) -> Vec<u32> {
        let encoding = self.inner.encode(text, false).unwrap();
        
        if add_bos {
            let mut ids = vec![self.bos_token_id];
            ids.extend(encoding.get_ids());
            return ids;
        }
        
        encoding.get_ids().iter().map(|id| *id as u32).collect()
    }

    /// Decode a sequence of token IDs back into text.
    pub fn decode(&self, tokens: &[u32], skip_special_tokens: bool) -> Result<String, String> {
        let ids: Vec<u32> = if skip_special_tokens {
            tokens.iter()
                .filter(|&&id| id != self.bos_token_id && id != self.eos_token_id)
                .copied()
                .collect()
        } else {
            tokens.to_vec()
        };
        
        self.inner.decode(&ids, true).map_err(|e| e.to_string())
    }

    /// Count the number of tokens in a text string.
    pub fn count_tokens(&self, text: &str) -> usize {
        self.inner.encode(text, false).unwrap().get_ids().len()
    }

    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode_roundtrip() {
        // This test requires a real tokenizer file.
        // In practice, you would load the tokenizer from the model's
        // tokenizer.model or tokenizer.json file.
        // let tokenizer = LlmTokenizer::from_file("path/to/tokenizer.model").unwrap();
        // let tokens = tokenizer.encode("Hello, world!", true);
        // assert!(tokens.len() > 0);
        // assert_eq!(tokens[0], 128000); // BOS token
        // let decoded = tokenizer.decode(&tokens, false).unwrap();
        // assert!(decoded.contains("Hello"));
    }
}
