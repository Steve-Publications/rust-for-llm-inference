use std::cell::Cell;
use cudarc::driver::{CudaContext, CudaSlice, CudaView};

/// Pre-allocate a GPU memory pool for tensor buffers.
///
/// One contiguous buffer is allocated up front; `allocate` hands out
/// non-owning views into it, so sub-allocations are instant and free of
/// repeated GPU malloc calls. The caller must not let sub-allocations
/// outlive `reset()` (they are not ref-counted individually).
pub struct GpuMemoryPool {
    buffer: CudaSlice<u8>,
    /// Total capacity in bytes.
    capacity: usize,
    /// Currently allocated bytes.
    used: Cell<usize>,
}

impl GpuMemoryPool {
    pub fn new(capacity: usize) -> Self {
        let context = CudaContext::new(0).unwrap();
        // Allocate a contiguous GPU buffer
        let buffer = context
            .default_stream()
            .alloc_zeros::<u8>(capacity)
            .unwrap();
        Self { buffer, capacity, used: Cell::new(0) }
    }

    /// Allocate a view of `size` bytes from the pool.
    pub fn allocate(&self, size: usize) -> Option<CudaView<'_, u8>> {
        if self.used.get() + size > self.capacity {
            return None;  // Pool exhausted
        }
        let offset = self.used.get();
        self.used.set(offset + size);
        Some(self.buffer.slice(offset..offset + size))
    }

    /// Reset the pool (make all sub-allocations available again).
    pub fn reset(&self) {
        self.used.set(0);
    }
}
