use opentelemetry::{global, Context, KeyValue};
use opentelemetry::global::BoxedTracer;
use opentelemetry::trace::{Span, SpanKind, TraceContextExt, Tracer};
use opentelemetry_sdk::trace::SdkTracerProvider;

/// Initialize OpenTelemetry with the OTLP exporter.
pub fn init_tracing() -> Result<BoxedTracer, Box<dyn std::error::Error>> {
    // Set up the provider with a batch span processor for efficiency.
    // Batch processing reduces network overhead by sending spans in groups
    // rather than one at a time. (The exporter sends traces to an OTLP
    // collector, which can forward them to Jaeger, Zipkin, etc.)
    let provider = SdkTracerProvider::builder()
        .with_batch_exporter(opentelemetry_otlp::SpanExporter::builder().build()?)
        .build();

    global::set_tracer_provider(provider);

    // The global tracer proxy resolves to the provider set above.
    Ok(global::tracer("llm-inference"))
}

/// Wrap an HTTP request handler with OpenTelemetry tracing.
async fn complete_with_tracing(
    State(state): State<AppState>,
    Json(req): Json<CompletionRequest>,
) -> Result<Json<CompletionResponse>, (StatusCode, String)> {
    let tracer = global::tracer("llm-inference");

    // Create a span for this request.
    let mut span = tracer
        .span_builder("complete_request")
        .with_kind(SpanKind::Server)
        .start(&tracer);

    span.add_event("request_received", vec![
        KeyValue::new("prompt_length", req.prompt.len() as i64),
    ]);

    let request_start = std::time::Instant::now();

    // Tokenize (as a child span of the request span).
    {
        // BoxedSpan is not Clone, so we link the child to the parent via the
        // parent's SpanContext (a cheap copy) instead of moving the span
        // into a Context.
        let parent_cx = Context::current()
            .with_remote_span_context(span.span_context().clone());
        let mut tokenize_span = tracer
            .span_builder("tokenize")
            .start_with_context(&tracer, &parent_cx);

        let tokens = state.engine.tokenizer.encode(&req.prompt, true);
        tokenize_span.add_event("tokenization_complete", vec![
            KeyValue::new("token_count", tokens.len() as i64),
        ]);
    }
    
    // Generate (with sub-spans for prefill and decode).
    let result = state.engine.generate(
        &req.prompt,
        req.max_tokens,
        req.temperature,
        req.top_p,
    ).map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    
    let elapsed = request_start.elapsed();
    
    span.add_event("request_completed", vec![
        KeyValue::new("total_time_ms", elapsed.as_millis() as i64),
        KeyValue::new("tokens_generated", result.generated_tokens as i64),
    ]);
    
    span.end();
    
    Ok(Json(CompletionResponse {
        text: result.text,
        tokens: result.tokens,
        prompt_tokens: result.prompt_tokens,
        generated_tokens: result.generated_tokens,
        total_time_ms: elapsed.as_millis() as f64,
    }))
}
