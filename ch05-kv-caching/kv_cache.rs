use candle_core::{Tensor, Device, DType, Result};

/// A simple contiguous KV cache for a single layer.
pub struct ContiguousKvCache {
    k: Tensor,  // [batch, heads, capacity, head_dim]
    v: Tensor,  // [batch, heads, capacity, head_dim]
    /// Number of tokens currently stored (0 <= write_pos <= capacity)
    write_pos: usize,
}

impl ContiguousKvCache {
    pub fn new(
        batch_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
        capacity: usize,
        device: &Device,
    ) -> Result<Self> {
        Ok(Self {
            k: Tensor::zeros((batch_size, num_kv_heads, capacity, head_dim), DType::F16, device)?,
            v: Tensor::zeros((batch_size, num_kv_heads, capacity, head_dim), DType::F16, device)?,
            write_pos: 0,
        })
    }

    /// Append new K and V tensors for the current position.
    /// The input tensors should have shape [batch, heads, 1, head_dim]
    /// and the same dtype as the cache (F16 here).
    pub fn append(&mut self, k_new: &Tensor, v_new: &Tensor) -> Result<()> {
        let (batch, heads, _, head_dim) = k_new.dims4()?;
        assert_eq!(heads, v_new.dim(1)?);
        assert_eq!(head_dim, v_new.dim(3)?);

        // Write to the current position. slice_assign takes one range per
        // dimension and returns the updated tensor (it is not in-place).
        let ranges = [
            0..batch,
            0..heads,
            self.write_pos..self.write_pos + 1,
            0..head_dim,
        ];
        self.k = self.k.slice_assign(&ranges, k_new)?;
        self.v = self.v.slice_assign(&ranges, v_new)?;

        self.write_pos += 1;
        Ok(())
    }

    /// Get the full cached K and V tensors up to write_pos.
    pub fn get_full(&self) -> Result<(Tensor, Tensor)> {
        let k_full = self.k.narrow(2, 0, self.write_pos)?;  // [:, :, :write_pos, :]
        let v_full = self.v.narrow(2, 0, self.write_pos)?;
        Ok((k_full, v_full))
    }

    /// Reset the cache (called when a request completes).
    pub fn reset(&mut self) {
        self.write_pos = 0;
    }
}
