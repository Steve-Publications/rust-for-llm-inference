use axum::{
    routing::{get, post},
    Json, Router,
    extract::State,
    response::sse::{Event, Sse},
};
use axum::http::StatusCode;
use serde::{Deserialize, Serialize};
use tokio::net::TcpListener;
use std::sync::Arc;
use futures_util::{Stream, StreamExt};

/// Request body for completions endpoint.
#[derive(Deserialize, Debug)]
pub struct CompletionRequest {
    pub prompt: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f64,
    #[serde(default = "default_top_p")]
    pub top_p: f64,
}

fn default_max_tokens() -> usize { 256 }
fn default_temperature() -> f64 { 0.7 }
fn default_top_p() -> f64 { 0.9 }

/// Non-streaming response.
#[derive(Serialize, Debug)]
pub struct CompletionResponse {
    pub text: String,
    pub tokens: Vec<u32>,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub total_time_ms: f64,
}

/// Application state shared across request handlers.
#[derive(Clone)]
pub struct AppState {
    pub engine: Arc<InferenceEngine>,
    pub scheduler: Arc<Scheduler>,
}

/// Handle a completion request (non-streaming).
pub async fn complete(
    State(state): State<AppState>,
    Json(req): Json<CompletionRequest>,
) -> Result<Json<CompletionResponse>, (StatusCode, String)> {
    let start = std::time::Instant::now();
    
    // Generate text synchronously
    let result = state.engine.generate(
        &req.prompt,
        req.max_tokens,
        req.temperature,
        req.top_p,
    ).map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    
    let elapsed = start.elapsed().as_millis() as f64;
    
    Ok(Json(CompletionResponse {
        text: result.text,
        tokens: result.tokens,
        prompt_tokens: result.prompt_tokens,
        generated_tokens: result.generated_tokens,
        total_time_ms: elapsed,
    }))
}

/// Handle a streaming completion request (Server-Sent Events).
pub async fn complete_stream(
    State(state): State<AppState>,
    Json(req): Json<CompletionRequest>,
) -> Result<Sse<impl Stream<Item = Result<Event, axum::Error>>>, (StatusCode, String)> {
    let engine = state.engine.clone();
    let prompt = req.prompt.clone();
    let max_tokens = req.max_tokens;
    let temperature = req.temperature;
    let top_p = req.top_p;
    
    // Generate tokens one at a time, yielding each as a streaming event
    let stream = async_stream::stream! {
        let mut token_stream = engine.generate_stream(
            &prompt,
            max_tokens,
            temperature,
            top_p,
        );

        while let Some(chunk) = token_stream.next().await {
            match chunk {
                Ok(StreamChunk { text, token_id, is_final }) => {
                    yield Ok(Event::default()
                        .json_data(StreamChunk {
                            text,
                            token_id,
                            is_final,
                        })
                        .unwrap());

                    if is_final {
                        break;
                    }
                }
                Err(e) => {
                    // Propagate engine errors to the client as an SSE error
                    yield Err(axum::Error::new(std::io::Error::new(
                        std::io::ErrorKind::Other, e
                    )));
                    break;
                }
            }
        }
    };
    
    Ok(Sse::new(stream))
}

/// Health check endpoint.
pub async fn health() -> StatusCode {
    StatusCode::OK
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load model and initialize engine
    let device = candle_core::Device::new_cuda(0)?;
    let config = ModelConfig::llama3_8b();
    let transformer = Transformer::load(&config, "model.safetensors", &device)?;
    let tokenizer = LlmTokenizer::from_file("tokenizer.model")?;

    let engine = InferenceEngine::new(transformer, tokenizer, device);
    let scheduler = Scheduler::new(
        1024,   // total blocks
        16,     // block size (tokens per block)
        4096,   // max prefill tokens
    );
    
    let state = AppState {
        engine: Arc::new(engine),
        scheduler: Arc::new(scheduler),
    };
    
    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/completions", post(complete))
        .route("/v1/completions/stream", post(complete_stream))
        .with_state(state);
    
    let listener = TcpListener::bind("0.0.0.0:8080").await?;
    println!("Server listening on http://0.0.0.0:8080");
    
    axum::serve(listener, app).await?;
    
    Ok(())
}
