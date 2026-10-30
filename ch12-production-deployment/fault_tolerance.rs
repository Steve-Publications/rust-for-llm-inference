use tracing::error;

/// Fault-tolerant version of the Chapter 11 `complete` handler.
/// A GPU out-of-memory failure is surfaced as 503 (retryable, possibly
/// on another replica) instead of a generic 500.
pub async fn complete(
    State(state): State<AppState>,
    Json(req): Json<CompletionRequest>,
) -> Result<Json<CompletionResponse>, (StatusCode, String)> {
    let start = std::time::Instant::now();
    let result = match state.engine.generate(
        &req.prompt,
        req.max_tokens,
        req.temperature,
        req.top_p,
    ) {
        Ok(result) => result,
        Err(e) if is_cuda_oom(&e) => {
            // Log and reject with 503 so the client can retry.
            error!("error" = %e, "GPU out of memory");
            return Err((
                StatusCode::SERVICE_UNAVAILABLE,
                "GPU memory exhausted. Try again shortly.".into(),
            ));
        }
        Err(e) => {
            error!("error" = %e, "Inference error");
            return Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string()));
        }
    };

    let elapsed = start.elapsed().as_millis() as f64;
    Ok(Json(CompletionResponse {
        text: result.text,
        tokens: result.tokens,
        prompt_tokens: result.prompt_tokens,
        generated_tokens: result.generated_tokens,
        total_time_ms: elapsed,
    }))
}

/// Returns true when the error is a CUDA out-of-memory failure.
///
/// Candle boxes the underlying `cudarc` error inside
/// `candle_core::Error::Cuda`, so we downcast to inspect the
/// driver error code.
pub fn is_cuda_oom(e: &candle_core::Error) -> bool {
    match e {
        candle_core::Error::Cuda(err) => {
            err.downcast_ref::<candle_core::cuda_backend::CudaError>()
                .map(|err| {
                    matches!(
                        err,
                        candle_core::cuda_backend::CudaError::Cuda(d)
                            if d.0
                                == cudarc::driver::sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY
                    )
                })
                .unwrap_or(false)
        }
        _ => false,
    }
}
