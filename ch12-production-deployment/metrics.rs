use prometheus::{
    register_histogram_with_registry,
    register_counter_with_registry,
    register_gauge_with_registry,
    register_histogram_vec_with_registry,
    Histogram, Counter, Gauge, HistogramVec, Registry,
};

/// Prometheus metrics for the inference engine.
pub struct InferenceMetrics {
    /// Label value for `latency_by_model` (from `new`).
    pub model_name: String,

    // Request counters
    pub requests_total: Counter,
    pub requests_failed: Counter,
    
    // Latency histograms (buckets in seconds)
    pub latency_seconds: Histogram,
    pub ttft_seconds: Histogram,       // Time to first token
    pub prefill_time_ms: Histogram,
    pub decode_time_ms: Histogram,
    
    // Throughput gauges
    pub tokens_per_second: Gauge,
    pub requests_per_second: Gauge,
    
    // Resource utilization gauges
    pub kv_cache_utilization: Gauge,   // 0.0 to 1.0 (fraction of blocks used)
    pub gpu_memory_bytes_used: Gauge,
    pub gpu_memory_bytes_total: Gauge,
    pub queue_depth: Gauge,            // Pending requests
    pub active_requests: Gauge,        // Currently processing
    
    // Per-model histograms (for multi-model deployments)
    pub latency_by_model: HistogramVec,
}

impl InferenceMetrics {
    pub fn new(registry: &Registry, model_name: &str) -> Self {
        let label_names = vec!["model"];
        
        Self {
            model_name: model_name.to_string(),
            requests_total: register_counter_with_registry!(
                "inference_requests_total",
                "Total number of inference requests",
                registry
            ).unwrap(),
            
            requests_failed: register_counter_with_registry!(
                "inference_requests_failed_total",
                "Total number of failed inference requests",
                registry
            ).unwrap(),
            
            latency_seconds: register_histogram_with_registry!(
                "inference_latency_seconds",
                "End-to-end inference latency",
                vec![0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
                registry
            ).unwrap(),
            
            ttft_seconds: register_histogram_with_registry!(
                "time_to_first_token_seconds",
                "Time from request receipt to first output token",
                vec![0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
                registry
            ).unwrap(),
            
            prefill_time_ms: register_histogram_with_registry!(
                "prefill_time_ms",
                "Time to process the prompt (prefill phase)",
                vec![10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0],
                registry
            ).unwrap(),
            
            decode_time_ms: register_histogram_with_registry!(
                "decode_time_ms",
                "Time per decoded token",
                vec![1.0, 5.0, 10.0, 20.0, 50.0],
                registry
            ).unwrap(),
            
            tokens_per_second: register_gauge_with_registry!(
                "tokens_per_second",
                "Current throughput in tokens per second",
                registry
            ).unwrap(),
            
            requests_per_second: register_gauge_with_registry!(
                "requests_per_second",
                "Current request rate",
                registry
            ).unwrap(),
            
            kv_cache_utilization: register_gauge_with_registry!(
                "kv_cache_utilization",
                "KV cache utilization as a fraction (0.0 to 1.0)",
                registry
            ).unwrap(),
            
            gpu_memory_bytes_used: register_gauge_with_registry!(
                "gpu_memory_bytes_used",
                "Current GPU memory usage in bytes",
                registry
            ).unwrap(),
            
            gpu_memory_bytes_total: register_gauge_with_registry!(
                "gpu_memory_bytes_total",
                "Total available GPU memory in bytes",
                registry
            ).unwrap(),
            
            queue_depth: register_gauge_with_registry!(
                "queue_depth",
                "Number of requests waiting for processing",
                registry
            ).unwrap(),
            
            active_requests: register_gauge_with_registry!(
                "active_requests",
                "Number of requests currently being processed",
                registry
            ).unwrap(),
            
            latency_by_model: register_histogram_vec_with_registry!(
                "inference_latency_by_model_seconds",
                "End-to-end inference latency by model",
                label_names.as_slice(),
                vec![0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
                registry
            ).unwrap(),
        }
    }
    
    /// Record metrics after a request completes.
    pub fn record_request(&self, latency_s: f64, tokens: usize, 
                          prefill_ms: f64, decode_ms: f64, ttft_s: f64) {
        self.requests_total.inc();
        self.latency_seconds.observe(latency_s);
        self.prefill_time_ms.observe(prefill_ms);
        self.decode_time_ms.observe(decode_ms);
        self.ttft_seconds.observe(ttft_s);
        
        if tokens > 0 && latency_s > 0.0 {
            self.tokens_per_second.set(tokens as f64 / latency_s);
        }
        
        self.latency_by_model
            .with_label_values(&[&self.model_name])
            .observe(latency_s);
    }
    
    /// Record a failed request.
    pub fn record_failure(&self, reason: &str) {
        self.requests_failed.inc();
        // In production, use a labeled counter:
        // self.request_failures.with_label_values(&[reason]).inc();
    }
}
