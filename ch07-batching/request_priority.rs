use std::cmp::Ordering;

/// Priority level for a pending request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Priority {
    /// Interactive chat: latency-critical (TTFT < 200ms).
    Interactive,
    /// Batch processing: throughput-critical.
    Throughput,
    /// Standard priority.
    Normal,
}

/// A priority-aware pending request.
#[derive(Debug)]
pub struct PrioritizedRequest {
    pub request: PendingRequest,
    pub priority: Priority,
    pub arrival_time: std::time::Instant,
    /// Dynamic priority score (higher = more urgent).
    pub score: f64,
}

impl PrioritizedRequest {
    /// Compute the dynamic priority score based on age and request type.
    pub fn recompute_score(&mut self) {
        let age_seconds = self.arrival_time.elapsed().as_secs_f64();
        
        let base = match self.priority {
            Priority::Interactive => 100.0,
            Priority::Throughput => 50.0,
            Priority::Normal => 25.0,
        };
        
        // Age bonus: older requests get a small priority boost to prevent starvation
        let age_bonus = age_seconds * 2.0;
        
        self.score = base + age_bonus;
    }
}

impl PartialEq for PrioritizedRequest {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}

// `Ord` requires `Eq`. Scores are f64 (not `Eq`), so implement `Eq`
// trivially - comparisons always go through the (total) `Ord` impl.
impl Eq for PrioritizedRequest {}

impl PartialOrd for PrioritizedRequest {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PrioritizedRequest {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap is a MAX-heap: the element that compares "greatest"
        // is popped first. Higher score = higher priority, so no reverse.
        self.score.partial_cmp(&other.score).unwrap_or(Ordering::Equal)
    }
}
