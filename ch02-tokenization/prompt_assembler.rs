/// Assembles a chat prompt according to a model-specific template.
pub struct PromptAssembler {
    /// The chat template string with placeholders.
    template: String,
}

impl PromptAssembler {
    /// Create an assembler for Llama 3's chat template.
    pub fn llama3() -> Self {
        Self {
            template: "<|begin_of_text|>\n\
                       <|start_header_id|>system<|end_header_id|>\n\
                       {system_prompt}<|eot_id|>\n\
                       <|start_header_id|>user<|end_header_id|>\n\
                       {user_message}<|eot_id|>\n\
                       <|start_header_id|>assistant<|end_header_id|>\n"
            .to_string(),
        }
    }

    /// Assemble a prompt from messages.
    pub fn assemble(&self, system_prompt: &str, user_message: &str) -> String {
        self.template
            .replace("{system_prompt}", system_prompt)
            .replace("{user_message}", user_message)
    }

    /// Count total tokens including template overhead.
    pub fn total_token_count(
        &self,
        tokenizer: &LlmTokenizer,
        system_prompt: &str,
        user_message: &str,
    ) -> usize {
        let prompt = self.assemble(system_prompt, user_message);
        // BOS is added during encoding, so we count +1 for it
        tokenizer.count_tokens(&prompt) + 1
    }
}
