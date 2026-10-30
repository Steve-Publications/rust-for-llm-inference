use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;

/// Launch the dequantize-and-matmul CUDA kernel (kernels/dequant_matmul.cu).
///
/// cudarc's launch builder is the safe counterpart of the raw `void*[]`
/// kernelParams convention: parameters are appended in the exact order of
/// the C signature (A, B, C, scales, M, N, K, block_size) and scalar
/// arguments are passed by reference and must outlive the launch call.
/// All slices must live on `context`'s device. `b` and `scales` hold raw
/// fp16 bits (`u16`); kernelParams passes device pointers, so the element
/// type only matters for sizing on the host.
pub fn launch_dequant_matmul(
    context: &Arc<CudaContext>,
    ptx: &str,
    a: &CudaSlice<u8>,       // [M x K_packed] 2-bit weights, 4 per byte
    b: &CudaSlice<u16>,      // [K x N] FP16 activations (raw half bits)
    c: &CudaSlice<f32>,      // [M x N] FP32 output
    scales: &CudaSlice<u16>, // [M/block_size x N] scale factors (raw half bits)
    m: usize,
    n: usize,
    k: usize,
    block_size: usize,
) -> Result<(), String> {
    let block_dim = (16u32, 16u32, 1);
    let grid_dim = (
        ((n as u32) + 15) / 16,
        ((m as u32) + 15) / 16,
        1,
    );

    // Load the kernel compiled by cudaforge at build time (see build.rs).
    let module = context
        .load_module(Ptx::from_src(ptx))
        .map_err(|e| e.to_string())?;
    let func = module
        .load_function("dequant_matmul_kernel")
        .map_err(|e| e.to_string())?;

    // Build the argument list in the exact order of the C signature:
    // A, B, C, scales, M, N, K, block_size.
    let (m32, n32, k32, bs32) = (m as i32, n as i32, k as i32, block_size as i32);
    let stream = context.default_stream();
    let mut launch = stream.launch_builder(&func);
    launch.arg(a).arg(b).arg(c).arg(scales);
    launch.arg(&m32).arg(&n32).arg(&k32).arg(&bs32);
    unsafe {
        launch
            .launch(LaunchConfig {
                block_dim,
                grid_dim,
                shared_mem_bytes: 0,
            })
            .map_err(|e| e.to_string())?;
    }
    context.synchronize().map_err(|e| e.to_string())?;

    Ok(())
}
