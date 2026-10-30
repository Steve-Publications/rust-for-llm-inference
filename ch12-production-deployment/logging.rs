use tracing::{info, Span};
use tracing_subscriber::{EnvFilter, Layer, prelude::*};

/// Generate a unique request ID.
fn generate_request_id() -> String {
    format!("req-{:012x}", rand::random::<u64>())
}

/// Set up structured logging with correlation IDs.
pub fn setup_logging() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        EnvFilter::new("info,llm_rs=debug")
    });
    
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::fmt::layer()
                .json()
                .with_current_span(true)
                .with_span_list(true)
                .with_filter(filter),
        )
        .init();
}

/// Middleware to attach a request ID to every request.
async fn request_id_middleware(
    req: axum::http::Request<axum::body::Body>,
    next: axum::middleware::Next,
) -> axum::http::Response<axum::body::Body> {
    let request_id = generate_request_id();
    
    // Attach the request ID to the tracing span.
    Span::current().record("request_id", &tracing::field::display(&request_id));
    
    info!("method" = req.method().to_string(),
          "path" = req.uri().to_string(),
          "request_id" = &request_id,
          "Incoming request");

    let response = next.run(req).await;

    // Log the response.
    info!("status" = response.status().as_u16(),
          "request_id" = &request_id,
          "Response sent");
    
    response
}

// Example structured log output:
// {"timestamp":"2026-01-15T14:32:01.123Z","level":"INFO",
//  "message":"Request completed","request_id":"req-a1b2c3d4e5f6",
//  "prompt_tokens":45,"generated_tokens":14,
//  "total_time_ms":224.8,"tokens_per_second":62.3}
