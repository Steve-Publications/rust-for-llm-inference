# llm-rs: Code from *Building High-Performance LLM Inference Engine from Scratch*

This folder contains every code listing from the book, extracted verbatim and
organized by chapter. There is one file per fenced code block, named after
the section it appears in; the book remains the source of truth.

Notes for readers:

- `.rs`, `.cu`, `.py`, `.toml`, `.yaml`, `.json`, `.sh` and `Dockerfile*` files are
  the book's runnable or configuration listings.
- `.txt` files hold the book's ASCII diagrams, equations and pseudocode
  (they appear inside code fences in the text but are not programs).
- Most Rust files are excerpts built around candle-core APIs from the chapters;
  they illustrate the surrounding discussion and are not a standalone crate.
  See Appendix A for how to obtain and run the complete `llm-rs` project
  (`clone_source.sh`, `quickstart.sh`).

## Chapter 1 - The Transformer Architecture

- `attention-tensor-shapes.txt` - Q/K/V projection shapes (figure, book line 141, 3 lines)
- `attention-equation.txt` - scaled dot-product attention formula (figure, book line 149, 1 line)
- `transformer-anatomy.txt` - layer-by-layer transformer anatomy diagram (figure, book line 187, 18 lines)
- `rope-pair-rotation.txt` - RoPE dimension-pair rotation rules (figure, book line 212, 5 lines)
- `rmsnorm-equation.txt` - RMSNorm formula (figure, book line 224, 1 line)
- `layernorm-equation.txt` - LayerNorm formula (figure, book line 231, 1 line)
- `rmsnorm-vs-layernorm.txt` - what RMSNorm drops from LayerNorm (figure, book line 236, 1 line)
- `swiglu-equation.txt` - SwiGLU feed-forward formula (figure, book line 256, 1 line)
- `prenorm-residual.txt` - pre-norm residual connections (figure, book line 267, 2 lines)
- `postnorm-residual.txt` - post-norm residual connections (figure, book line 273, 2 lines)
- `forward-pass-graph.txt` - forward pass as a computation graph (figure, book line 286, 49 lines)
- `model_config.rs` - ModelConfig hyperparameters with presets and tests (listing, book line 354, 105 lines)

## Chapter 2 - Tokenization and the Vocabulary

- `bpe-training-corpus.txt` - BPE training corpus example (figure, book line 494, 1 line)
- `tokenizer.rs` - Hugging Face tokenizers wrapper with tests (listing, book line 536, 118 lines)
- `llama3-chat-template.txt` - Llama 3 chat template (figure, book line 680, 6 lines)
- `mistral-chat-template.txt` - Mistral chat template (figure, book line 691, 1 line)
- `prompt_assembler.rs` - chat prompt assembly and token counting (listing, book line 701, 39 lines)

## Chapter 3 - Attention Mechanisms Deep Dive

- `scaled-dot-product-equation.txt` - scaled dot-product attention formula (figure, book line 768, 1 line)
- `attention-example-input.txt` - worked example: Q input matrix (figure, book line 780, 8 lines)
- `attention-example-scores.txt` - worked example: Q*K^T scores (figure, book line 792, 2 lines)
- `attention-example-scaling.txt` - worked example: scaling by 1/sqrt(d_k) (figure, book line 798, 3 lines)
- `attention-example-softmax.txt` - worked example: softmax weights (figure, book line 805, 2 lines)
- `scaled_dot_product_attention.rs` - scaled dot-product attention in candle (listing, book line 814, 29 lines)
- `multi-head-attention-equation.txt` - multi-head attention formula (figure, book line 854, 2 lines)
- `multi_head_attention.rs` - multi-head attention with head pruning (listing, book line 870, 225 lines)
- `gpu-memory-hierarchy.txt` - GPU memory hierarchy diagram (figure, book line 1106, 18 lines)
- `flashattention-pseudocode.txt` - FlashAttention tiling pseudocode (figure, book line 1131, 18 lines)
- `causal-mask.txt` - causal mask matrix (figure, book line 1162, 5 lines)
- `rope-rotation-pseudocode.txt` - RoPE rotation pseudocode (figure, book line 1180, 5 lines)
- `rope-relative-offset-equation.txt` - RoPE relative-position property (figure, book line 1190, 1 line)
- `rope.rs` - RoPE implementation in candle (listing, book line 1198, 62 lines)

## Chapter 4 - Building the Core Inference Loop

- `Cargo.toml` - llm-rs crate manifest (listing, book line 1286, 20 lines)
- `linear.rs` - linear layer y = x @ W^T + b (listing, book line 1317, 26 lines)
- `rms_norm.rs` - RMSNorm forward pass (listing, book line 1350, 20 lines)
- `swiglu.rs` - SwiGLU feed-forward layer (listing, book line 1377, 36 lines)
- `transformer-layer-graph.txt` - transformer layer dataflow diagram (figure, book line 1422, 24 lines)
- `transformer_layer.rs` - complete transformer layer composition (listing, book line 1449, 235 lines)
- `prefill.rs` - prefill: full prompt in one pass (listing, book line 1693, 19 lines)
- `decode.rs` - decode: single token with KV cache (listing, book line 1717, 16 lines)
- `generate.rs` - autoregressive generation loop (listing, book line 1738, 55 lines)
- `bench_layer.rs` - Criterion benchmark for a single layer (listing, book line 1802, 36 lines)

## Chapter 5 - KV Caching and Autoregressive Optimization

- `attention-complexity-equation.txt` - O(n^2) attention cost derivation (figure, book line 1887, 1 line)
- `kv-cache-size-equation.txt` - KV cache size formula (figure, book line 1893, 1 line)
- `kv-cache-size-example-8b.txt` - KV cache size for an 8B model (figure, book line 1906, 3 lines)
- `kv-cache-size-example-70b.txt` - KV cache size for a 70B model (figure, book line 1914, 2 lines)
- `kv-cache-tensor-shapes.txt` - per-layer K/V cache shapes (figure, book line 1927, 2 lines)
- `kv_cache.rs` - contiguous KV cache implementation (listing, book line 1934, 60 lines)
- `paged-memory-layout.txt` - paged physical memory diagram (figure, book line 2007, 16 lines)
- `block-table-example.txt` - logical-to-physical block table (figure, book line 2032, 10 lines)
- `paged_attention.rs` - PagedAttention block manager and kernel (listing, book line 2049, 263 lines)
- `cache-eviction-budget.txt` - per-layer cache budget and eviction math (figure, book line 2322, 3 lines)

## Chapter 6 - Quantization for Memory and Speed

- `fp32-bit-layout-equation.txt` - FP32 bit layout (figure, book line 2361, 1 line)
- `fp8_recipe.py` - Transformer Engine FP8 configuration (listing, book line 2377, 7 lines)
- `quantize-dequantize-equations.txt` - symmetric/asymmetric quantization equations (figure, book line 2395, 5 lines)
- `absmax-quantization-equation.txt` - absmax INT8 quantization (figure, book line 2404, 2 lines)
- `gguf-file-layout.txt` - GGUF file layout diagram (figure, book line 2429, 24 lines)
- `dequant_int4.rs` - INT4 dequantization (listing, book line 2480, 27 lines)
- `dequant_mul_int4.rs` - fused dequantize-and-multiply (listing, book line 2514, 29 lines)

## Chapter 7 - Batching Strategies

- `static-batching-timeline.txt` - static batching timeline (figure, book line 2586, 9 lines)
- `dynamic-batching-timeline.txt` - dynamic batching timeline (figure, book line 2602, 10 lines)
- `continuous_batching.rs` - continuous batching scheduler (listing, book line 2625, 162 lines)
- `continuous-batching-paged-attention.txt` - batching + PagedAttention interplay (figure, book line 2796, 35 lines)
- `request_priority.rs` - request priority ordering (listing, book line 2874, 64 lines)
- `priority_scheduler.rs` - priority queue scheduler with preemption (listing, book line 2943, 246 lines)
- `work_stealing_prefill.rs` - work-stealing multi-threaded prefill (listing, book line 3196, 124 lines)
- `scheduler_simulation.rs` - scheduling simulation and benchmarking (listing, book line 3332, 186 lines)
- `radix-tree-prefix-cache.txt` - RadixAttention prefix cache diagram (figure, book line 3527, 10 lines)

## Chapter 8 - CPU Optimization Techniques

- `simd_sum.rs` - nightly portable_simd vector sum (listing, book line 3587, 21 lines)
- `matmul_unrolled.rs` - 4-way loop-unrolled matmul (listing, book line 3618, 51 lines)
- `matmul_avx2.rs` - explicit AVX2/FMA matmul (listing, book line 3676, 72 lines)
- `cache-lines-diagram.txt` - multi-core cache line diagram (figure, book line 3757, 23 lines)
- `matmul-loop-order.txt` - loop order and cache locality (figure, book line 3787, 16 lines)
- `rayon_matmul.rs` - Rayon parallel matmul (listing, book line 3814, 22 lines)
- `bench_matmul.rs` - Criterion benchmark with scaling study (listing, book line 3876, 37 lines)

## Chapter 9 - GPU Acceleration with CUDA and ROCm

- `cuda_device.rs` - creating a CUDA device and tensor (listing, book line 3973, 2 lines)
- `build.rs` - cudarc build script with NVRTC (listing, book line 3988, 32 lines)
- `cuda_backend.rs` - CUDA backend for llm-rs (listing, book line 4025, 51 lines)
- `dequant_matmul.cu` - INT2 dequantize-and-matmul CUDA kernel (listing, book line 4093, 68 lines)
- `dequant_matmul_launch.rs` - Rust host launcher for the kernel (listing, book line 4166, 61 lines)
- `flash_attention.cu` - FlashAttention-2 CUDA kernel structure (listing, book line 4234, 82 lines)
- `fused_rmsnorm_gelu.cu` - fused RMSNorm + GELU kernel (listing, book line 4328, 30 lines)
- `gpu_memory_pool.rs` - GPU tensor memory pool (listing, book line 4369, 43 lines)
- `nsight-timeline.txt` - Nsight timeline diagram (figure, book line 4425, 14 lines)
- `cuda_profiling.rs` - instrumented kernels for profiling (listing, book line 4448, 51 lines)

## Chapter 10 - Model Formats and Serialization

- `safetensors-file-layout.txt` - safetensors file layout diagram (figure, book line 4553, 20 lines)
- `safetensors_metadata.rs` - parsing safetensors headers (listing, book line 4588, 25 lines)
- `weight_layout.rs` - training-to-inference layout transform (listing, book line 4643, 27 lines)
- `gguf_loader.rs` - GGUF model loader (listing, book line 4677, 176 lines)

## Chapter 11 - Distributed Inference and Serving

- `tensor-parallel-diagram.txt` - tensor parallelism diagram (figure, book line 4880, 28 lines)
- `tensor-parallel-split.txt` - tensor vs pipeline split semantics (figure, book line 4913, 4 lines)
- `tp-communication-cost-equation.txt` - TP communication cost formula (figure, book line 4922, 1 line)
- `tp-communication-cost-example.txt` - TP communication cost example (figure, book line 4927, 2 lines)
- `pipeline-parallel-diagram.txt` - pipeline parallelism diagram (figure, book line 4936, 6 lines)
- `pipeline-bubble-equation.txt` - pipeline bubble formula (figure, book line 4946, 1 line)
- `ring-allreduce-steps.txt` - ring all-reduce steps (figure, book line 4960, 9 lines)
- `nccl_allreduce.rs` - NCCL all-reduce wrapper (listing, book line 4981, 91 lines)
- `inference_engine.rs` - inference engine shared by handlers (listing, book line 5092, 160 lines)
- `server.rs` - axum server with SSE streaming (listing, book line 5257, 158 lines)

## Chapter 12 - Production Deployment and Monitoring

- `Dockerfile` - multi-stage CPU Docker build (listing, book line 5444, 39 lines)
- `Dockerfile.gpu` - GPU runtime image (listing, book line 5488, 7 lines)
- `k8s-deployment.yaml` - Kubernetes deployment (listing, book line 5500, 75 lines)
- `k8s-readiness-probe.yaml` - readiness probe fragment (listing, book line 5588, 7 lines)
- `k8s-hpa.yaml` - HorizontalPodAutoscaler (listing, book line 5602, 41 lines)
- `metrics.rs` - Prometheus metrics implementation (listing, book line 5670, 161 lines)
- `grafana-dashboard.json` - Grafana dashboard panels (listing, book line 5857, 29 lines)
- `tracing.rs` - OpenTelemetry distributed tracing (listing, book line 5893, 82 lines)
- `trace-waterfall.txt` - trace waterfall diagram (figure, book line 5980, 41 lines)
- `logging.rs` - structured logging with correlation IDs (listing, book line 6036, 55 lines)
- `prometheus-alerts.yaml` - alerting rules (listing, book line 6104, 26 lines)
- `fault_tolerance.rs` - fault-tolerant completion handler (listing, book line 6139, 63 lines)
- `check_cuda_macro.rs` - check_cuda! error macro (listing, book line 6207, 10 lines)
- `k8s-readiness-probe-tuning.yaml` - probe tuning for failover (listing, book line 6228, 7 lines)
- `startup_main.rs` - fail-fast server startup (listing, book line 6240, 17 lines)
- `k8s-shadow-deployment.yaml` - A/B shadow deployment (listing, book line 6268, 27 lines)
- `canary-rollout-plan.txt` - canary rollout schedule (figure, book line 6300, 4 lines)
- `ab_bucket_hash.rs` - deterministic A/B bucket hashing (listing, book line 6309, 15 lines)

## Appendix A - Benchmarking Methodology

- `bench_attention.sh` - single-layer attention benchmark commands (listing, book line 6432, 5 lines)
- `bench_full_inference.sh` - full inference benchmark commands (listing, book line 6448, 8 lines)
- `clone_source.sh` - obtaining the llm-rs source (listing, book line 6467, 3 lines)
- `quickstart.sh` - quick start: test the crate (listing, book line 6475, 3 lines)
- `train_and_generate.sh` - train the tiny model and generate text (listing, book line 6486, 9 lines)
- `demo_generate.sh` - untrained demo generation (listing, book line 6506, 1 line)
- `command_reference.sh` - CLI command reference (listing, book line 6512, 14 lines)
