// build.rs
use cudaforge::KernelBuilder;
use std::path::PathBuf;

fn main() {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());

    // Compile CUDA kernels from .cu files to PTX. The CUDA driver
    // JIT-compiles the PTX for the host GPU the first time each
    // kernel is loaded at runtime.
    let ptx = KernelBuilder::new()
        .source_files([
            "kernels/matmul.cu",
            "kernels/flash_attn.cu",
            "kernels/rmsnorm_gelu.cu",
            "kernels/dequant_matmul.cu",
        ])
        .arg("-std=c++17")
        .build_ptx()
        .unwrap();

    // Embed each kernel's PTX as a string constant in a Rust file,
    // e.g. pub const MATMUL: &str =
    //     include_str!(concat!(env!("OUT_DIR"), "/matmul.ptx"));
    ptx.write(&out_dir.join("ptx.rs")).unwrap();

    // Link against the CUDA runtime (nvcc installs under CUDA_ROOT,
    // typically /usr/local/cuda, and the libraries live in lib64/)
    let cuda_root = std::env::var("CUDA_ROOT").unwrap_or_else(|_| "/usr/local/cuda".to_string());
    println!("cargo:rustc-link-search=native={}/lib64", cuda_root);
    println!("cargo:rustc-link-lib=dylib=cudart");
}
