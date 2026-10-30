use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;

/// CUDA matmul kernel: C = A × B
/// 
/// A: [M x K], B: [K x N], C: [M x N] (row-major)
/// Each thread computes one element of C.
/// `ptx` is the PTX for matmul_kernel generated at build time (see build.rs).
pub fn launch_matmul_kernel(
    context: &Arc<CudaContext>,
    ptx: &str,
    a: &CudaSlice<f32>,
    b: &CudaSlice<f32>,
    m: usize,
    n: usize,
    k: usize,
) -> CudaSlice<f32> {
    let c = context.default_stream().alloc_zeros::<f32>(m * n).unwrap();
    
    let block_dim = (16u32, 16u32, 1);
    let grid_dim = (
        (n as u32 + block_dim.0 - 1) / block_dim.0,
        (m as u32 + block_dim.1 - 1) / block_dim.1,
        1,
    );
    
    // Load the kernel compiled from kernels/matmul.cu
    let module = context.load_module(Ptx::from_src(ptx)).unwrap();
    let func = module.load_function("matmul_kernel").unwrap();
    
    // Arguments must be passed in the exact order of the C signature:
    // (a, b, c, m, n, k). Scalar arguments are passed by reference and
    // must outlive the launch call, so bind them first.
    let (m32, n32, k32) = (m as i32, n as i32, k as i32);
    let stream = context.default_stream();
    let mut launch = stream.launch_builder(&func);
    launch.arg(a).arg(b).arg(&c);
    launch.arg(&m32).arg(&n32).arg(&k32);
    unsafe {
        launch.launch(LaunchConfig {
            block_dim,
            grid_dim,
            shared_mem_bytes: 0,
        }).unwrap();
    }
    
    context.synchronize().unwrap();
    c
}
