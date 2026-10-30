use std::collections::HashMap;

use candle_core::{Device, Result, Tensor};

/// Load model weights from safetensors files.
/// 
/// Handles multi-file checkpoints (e.g., model-00001-of-00003.safetensors).
pub fn load_safetensors(
    paths: &[&str],
    device: &Device,
) -> Result<HashMap<String, Tensor>> {
    let mut all_tensors = HashMap::new();
    
    for path in paths {
        // Load every tensor from this shard into a map
        let tensors: HashMap<String, Tensor> =
            candle_core::safetensors::load(path, device)?;
        
        for (name, tensor) in tensors {
            all_tensors.insert(name, tensor);
        }
    }
    
    Ok(all_tensors)
}
