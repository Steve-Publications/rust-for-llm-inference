// Check for CUDA errors after each kernel launch.
macro_rules! check_cuda {
    ($expr:expr) => {{
        let result = $expr;
        if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            return Err(format!("CUDA error {:?} at {}:{}", 
                result, file!(), line!()));
        }
    }};
}
