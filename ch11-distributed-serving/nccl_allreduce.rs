use singe_nccl_sys::*;

/// Initialize NCCL for multi-GPU communication.
pub struct NcclCommunicator {
    comm: ncclComm_t,
}

impl NcclCommunicator {
    /// Generate a unique communicator ID. Call this on rank 0 and share
    /// the bytes with every other rank (via a file, env var, MPI broadcast
    /// or any other side channel) BEFORE they call `new`.
    pub fn generate_unique_id() -> Result<ncclUniqueId, String> {
        let mut unique_id = ncclUniqueId::default();
        if unsafe { ncclGetUniqueId(&mut unique_id) } != ncclResult_t::ncclSuccess {
            return Err("NCCL get_unique_id failed".to_string());
        }
        Ok(unique_id)
    }

    /// Join the communicator. Every rank must pass the SAME `unique_id`
    /// (generated on rank 0 and shared), its own `rank`, and the total
    /// `nranks`. `rank 0` should run on device 0, `rank 1` on device 1,
    /// and so on (pin devices with cudaSetDevice or CUDA_VISIBLE_DEVICES).
    pub fn new(rank: usize, nranks: usize, unique_id: &ncclUniqueId) -> Result<Self, String> {
        unsafe {
            let mut comm = std::ptr::null_mut();

            let result = ncclCommInitRank(
                &mut comm,
                nranks as i32,
                *unique_id,
                rank as i32,
            );

            if result != ncclResult_t::ncclSuccess {
                return Err(format!("NCCL init failed: {:?}", result));
            }

            Ok(Self { comm })
        }
    }

    /// Perform an all-reduce operation.
    pub fn all_reduce(&self, send_buf: *const f32, recv_buf: *mut f32, count: usize) -> Result<(), String> {
        unsafe {
            let result = ncclAllReduce(
                send_buf as *const std::ffi::c_void,
                recv_buf as *mut std::ffi::c_void,
                count as u64,
                ncclDataType_t::ncclFloat32,
                ncclRedOp_t::ncclSum,
                self.comm,
                std::ptr::null_mut(),
            );

            if result != ncclResult_t::ncclSuccess {
                return Err(format!("NCCL all_reduce failed: {:?}", result));
            }

            Ok(())
        }
    }

    /// Perform an all-gather operation.
    pub fn all_gather(&self, send_buf: *const f32, recv_buf: *mut f32, count: usize) -> Result<(), String> {
        unsafe {
            let result = ncclAllGather(
                send_buf as *const std::ffi::c_void,
                recv_buf as *mut std::ffi::c_void,
                count as u64,
                ncclDataType_t::ncclFloat32,
                self.comm,
                std::ptr::null_mut(),
            );

            if result != ncclResult_t::ncclSuccess {
                return Err(format!("NCCL all_gather failed: {:?}", result));
            }

            Ok(())
        }
    }
}

impl Drop for NcclCommunicator {
    fn drop(&mut self) {
        unsafe {
            ncclCommDestroy(self.comm);
        }
    }
}
