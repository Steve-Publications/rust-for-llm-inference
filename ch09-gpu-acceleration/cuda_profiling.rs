use std::sync::Arc;

use cudarc::driver::sys::CUevent_flags;
use cudarc::driver::{CudaContext, CudaEvent};

/// Simple CUDA event-based profiler.
///
/// Events are created with timing enabled; `start`/`stop` record them on
/// the default stream and `elapsed_ms` reads the GPU-side duration.
pub struct CudaProfiler {
    context: Arc<CudaContext>,
    start_event: CudaEvent,
    end_event: CudaEvent,
}

impl CudaProfiler {
    pub fn new() -> Self {
        let context = CudaContext::new(0).unwrap();
        let start_event = context
            .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .unwrap();
        let end_event = context
            .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .unwrap();
        Self {
            context,
            start_event,
            end_event,
        }
    }

    /// Record the start event on the default stream.
    pub fn start(&self) {
        self.start_event
            .record(&self.context.default_stream())
            .unwrap();
    }

    /// Record the end event on the default stream and wait for it.
    pub fn stop(&self) {
        self.end_event
            .record(&self.context.default_stream())
            .unwrap();
        self.end_event.synchronize().unwrap();
    }

    /// GPU time between the start and end events, in milliseconds.
    pub fn elapsed_ms(&self) -> f32 {
        self.start_event.elapsed_ms(&self.end_event).unwrap()
    }
}
