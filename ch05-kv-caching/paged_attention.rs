use std::collections::{HashMap, VecDeque};
use candle_core::{Tensor, Device, DType, Error, Result};

/// A block is a fixed-size chunk of KV cache memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(usize);

/// The block table maps logical sequence positions to physical blocks.
/// Each sequence has its own block table.
#[derive(Debug)]
pub struct BlockTable {
    /// For each logical block index, the physical block ID.
    blocks: Vec<BlockId>,
}

impl BlockTable {
    pub fn new() -> Self {
        Self { blocks: Vec::new() }
    }

    pub fn add_block(&mut self, block_id: BlockId) {
        self.blocks.push(block_id);
    }

    pub fn get_physical_block(&self, logical_idx: usize) -> Option<BlockId> {
        self.blocks.get(logical_idx).copied()
    }

    pub fn len(&self) -> usize {
        self.blocks.len()
    }
}

/// A sequence identifier (one per active request).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SeqId(pub usize);

/// A page is a fixed-size chunk of KV cache memory.
/// Each page stores K and V for `page_size` tokens, across all heads and layers.
pub struct Page {
    /// Physical block ID (assigned by the allocator).
    block_id: BlockId,
    /// The sequence this page belongs to.
    seq_id: Option<SeqId>,
}

/// The page allocator manages free and allocated pages.
pub struct PageAllocator {
    /// Total number of pages available.
    total_pages: usize,
    /// Number of pages currently allocated.
    used_pages: usize,
    /// Free pages waiting to be assigned.
    free_queue: VecDeque<BlockId>,
}

impl PageAllocator {
    pub fn new(total_pages: usize) -> Self {
        let mut free_queue = VecDeque::with_capacity(total_pages);
        for i in 0..total_pages {
            free_queue.push_back(BlockId(i));
        }
        Self {
            total_pages,
            used_pages: 0,
            free_queue,
        }
    }

    /// Allocate a single page (returns None if no pages are available).
    pub fn allocate(&mut self) -> Option<BlockId> {
        self.free_queue.pop_front().map(|id| {
            self.used_pages += 1;
            id
        })
    }

    /// Free a previously allocated page.
    pub fn free(&mut self, block_id: BlockId) {
        self.free_queue.push_back(block_id);
        self.used_pages -= 1;
    }

    /// Number of pages still available.
    pub fn available(&self) -> usize {
        self.total_pages - self.used_pages
    }
}

/// The Paged KV Cache manages block-level allocation for all layers.
pub struct PagedKvCache {
    allocator: PageAllocator,
    /// Per-layer physical buffers: [layer][block_id] -> (k_tensor, v_tensor)
    k_buffers: Vec<HashMap<BlockId, Tensor>>,
    v_buffers: Vec<HashMap<BlockId, Tensor>>,
    block_size: usize,  // tokens per block
}

impl PagedKvCache {
    pub fn new(
        num_layers: usize,
        batch_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
        block_size: usize,
        total_blocks: usize,
        device: &Device,
    ) -> Result<Self> {
        let allocator = PageAllocator::new(total_blocks);
        
        // Pre-allocate physical buffers for all blocks across all layers
        let mut k_buffers = Vec::with_capacity(num_layers);
        let mut v_buffers = Vec::with_capacity(num_layers);
        
        for _ in 0..num_layers {
            let mut k_bufs = HashMap::new();
            let mut v_bufs = HashMap::new();
            
            for block_id in 0..total_blocks {
                let bid = BlockId(block_id);
                // Each block: [batch, heads, block_size, head_dim]
                let k = Tensor::zeros(
                    (batch_size, num_kv_heads, block_size, head_dim),
                    DType::F16,
                    device,
                )?;
                let v = Tensor::zeros(
                    (batch_size, num_kv_heads, block_size, head_dim),
                    DType::F16,
                    device,
                )?;
                k_bufs.insert(bid, k);
                v_bufs.insert(bid, v);
            }
            
            k_buffers.push(k_bufs);
            v_buffers.push(v_bufs);
        }
        
        Ok(Self {
            allocator,
            k_buffers,
            v_buffers,
            block_size,
        })
    }

    /// Allocate blocks for a new sequence.
    /// Returns the block table mapping logical positions to physical blocks.
    pub fn allocate_blocks(&mut self, seq_id: SeqId) -> Option<BlockTable> {
        let mut block_table = BlockTable::new();
        
        // Allocate as many blocks as needed (we'll grow dynamically)
        if let Some(block_id) = self.allocator.allocate() {
            block_table.add_block(block_id);
            Some(block_table)
        } else {
            None  // No free blocks available
        }
    }

    /// Append new K and V data to the sequence's blocks.
    /// `k_data` and `v_data` should be [batch, heads, 1, head_dim] and the
    /// same dtype as the buffers (F16); they are written at position
    /// `pos_in_block` (0 <= pos_in_block < block_size) within the block.
    pub fn append_to_blocks(
        &mut self,
        layer: usize,
        block_id: BlockId,
        pos_in_block: usize,
        k_data: &Tensor,
        v_data: &Tensor,
    ) -> Result<()> {
        let (batch, heads, _, head_dim) = k_data.dims4()?;
        let ranges = [
            0..batch,
            0..heads,
            pos_in_block..pos_in_block + 1,
            0..head_dim,
        ];
        let k_buf = self.k_buffers[layer]
            .get_mut(&block_id)
            .ok_or_else(|| Error::Msg("Block not found".to_string()))?;
        *k_buf = k_buf.slice_assign(&ranges, k_data)?;
        let v_buf = self.v_buffers[layer]
            .get_mut(&block_id)
            .ok_or_else(|| Error::Msg("Block not found".to_string()))?;
        *v_buf = v_buf.slice_assign(&ranges, v_data)?;
        Ok(())
    }

    /// Gather K and V tensors for a sequence across all physical blocks.
    /// 
    /// This is called during the attention forward pass to assemble the full
    /// key and value tensors from scattered block buffers.
    pub fn gather_sequence(
        &self,
        layer: usize,
        block_table: &BlockTable,
    ) -> Result<(Tensor, Tensor)> {
        if block_table.blocks.is_empty() {
            return Err(Error::Msg("Empty block table".to_string()));
        }
        
        // Gather individual block tensors and concatenate along the sequence dimension.
        // Each block is [batch, heads, block_size, head_dim].
        let first_block = self.k_buffers[layer]
            .get(&block_table.blocks[0])
            .ok_or_else(|| Error::Msg("First block not found".to_string()))?;
        
        let batch_size = first_block.dim(0)?;
        let num_kv_heads = first_block.dim(1)?;
        let head_dim = first_block.dim(3)?;
        let total_tokens = block_table.len() * self.block_size;
        
        // Allocate contiguous output tensors
        let mut k_full = Tensor::zeros(
            (batch_size, num_kv_heads, total_tokens, head_dim),
            DType::F16,
            first_block.device(),
        )?;
        let mut v_full = Tensor::zeros(
            (batch_size, num_kv_heads, total_tokens, head_dim),
            DType::F16,
            first_block.device(),
        )?;
        
        // Copy data from each block into the correct position
        for (block_idx, &block_id) in block_table.blocks.iter().enumerate() {
            let k_block = self.k_buffers[layer].get(&block_id)
                .ok_or_else(|| Error::Msg(format!("Block {} not found", block_id.0)))?;
            let v_block = self.v_buffers[layer].get(&block_id)
                .ok_or_else(|| Error::Msg(format!("Block {} not found", block_id.0)))?;
            
            let start_pos = block_idx * self.block_size;
            let end_pos = (start_pos + self.block_size).min(total_tokens);
            let copy_len = end_pos - start_pos;
            
            // Copy the relevant slice of each block into k_full and v_full:
            // block [:, :, :copy_len, :] -> output [:, :, start_pos:end_pos, :].
            // slice_assign needs one range per dim and returns the updated tensor.
            let src_k = k_block.narrow(2, 0, copy_len)?;
            let src_v = v_block.narrow(2, 0, copy_len)?;
            let ranges = [
                0..batch_size,
                0..num_kv_heads,
                start_pos..start_pos + copy_len,
                0..head_dim,
            ];
            k_full = k_full.slice_assign(&ranges, &src_k)?;
            v_full = v_full.slice_assign(&ranges, &src_v)?;
        }
        
        Ok((k_full, v_full))
    }

    /// Free blocks for a completed sequence.
    pub fn free_sequence(&mut self, block_table: &BlockTable) {
        for block_id in &block_table.blocks {
            self.allocator.free(*block_id);
        }
    }
}
