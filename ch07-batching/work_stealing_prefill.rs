use rayon::scope;
use rayon::current_thread_index;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::sync::{Arc, Mutex};

/// A work-stealing prefill task.
#[derive(Debug)]
pub struct PrefillTask {
    pub seq_id: SeqId,
    pub tokens: Vec<u32>,
    pub priority: Priority,
    pub max_tokens: usize,
    pub temperature: f64,
    pub top_p: f64,
}

/// A work-stealing scheduler for prefill tasks.
///
/// Each worker thread has its own deque of tasks. When a thread runs out
/// of work, it steals from the back of another thread's deque.
///
/// The deques are mutex-protected here for clarity; a production engine
/// would use a lock-free Chase-Lev deque (which is exactly what Rayon uses
/// for its own internal task stealing).
pub struct WorkStealingScheduler {
    /// Per-thread task deques (one per CPU core).
    deques: Vec<Arc<Mutex<VecDeque<PrefillTask>>>>,
    /// Number of worker threads.
    num_workers: usize,
    /// Round-robin counter for task distribution.
    next_worker: AtomicUsize,
}

impl WorkStealingScheduler {
    pub fn new(num_workers: usize) -> Self {
        Self {
            deques: (0..num_workers)
                .map(|_| Arc::new(Mutex::new(VecDeque::new())))
                .collect(),
            num_workers,
            next_worker: AtomicUsize::new(0),
        }
    }

    /// Add a task to the scheduler. Tasks are distributed round-robin.
    pub fn submit(&self, task: PrefillTask) {
        // (self.deques.len() % self.num_workers) would always be 0 -
        // use an atomic counter so tasks rotate across workers.
        let worker_idx = self.next_worker.fetch_add(1, AtomicOrdering::Relaxed) % self.num_workers;
        let mut deque = self.deques[worker_idx].lock().unwrap();
        deque.push_back(task);
    }

    /// Take the next task for `worker`: from the front of its own deque
    /// first (best cache locality), then from the back of any other deque
    /// that still has work.
    fn take_task(&self, worker: usize) -> Option<PrefillTask> {
        if let Some(task) = self.deques[worker].lock().unwrap().pop_front() {
            return Some(task);
        }

        for (idx, deque) in self.deques.iter().enumerate() {
            if idx == worker {
                continue;
            }
            // Stealing takes from the *back*: the owner pops from the front,
            // so owner and stealer rarely touch the same element.
            if let Some(task) = deque.lock().unwrap().pop_back() {
                return Some(task);
            }
        }

        None
    }

    /// Process all pending tasks on Rayon's thread pool.
    ///
    /// Each worker drains its own deque and steals from the others when it
    /// runs dry, so long prompts cannot leave one thread while the rest
    /// idle. Results come back in completion order, not submission order,
    /// because the workers run concurrently.
    pub fn process_all<F, R>(&self, process_fn: F) -> Vec<(SeqId, R)>
    where
        // The closure is shared by every worker thread, so it has to cross
        // thread boundaries (`Send + Sync`), and so does its result.
        F: Fn(&PrefillTask) -> R + Send + Sync,
        R: Send,
    {
        let results: Arc<Mutex<Vec<(SeqId, R)>>> = Arc::new(Mutex::new(Vec::new()));
        // Shared by every worker: each job gets its own `Arc` handle.
        let process_fn = Arc::new(process_fn);

        scope(|scope| {
            for _ in 0..self.num_workers {
                let results = Arc::clone(&results);
                let process_fn = Arc::clone(&process_fn);
                scope.spawn(move |_| {
                    // Rayon tells us which pool thread runs this job, so each
                    // job starts on the deque assigned to that worker.
                    let own = current_thread_index().unwrap_or(0) % self.num_workers;
                    // `take_task` only returns None once every deque - this
                    // worker's own and all the others - is empty.
                    while let Some(task) = self.take_task(own) {
                        let seq_id = task.seq_id;
                        let output = process_fn.as_ref()(&task);
                        results.lock().unwrap().push((seq_id, output));
                    }
                });
            }
        });

        // `scope` joins all spawned jobs before returning.
        let mut results = results.lock().unwrap();
        std::mem::take(&mut *results)
    }

    /// Get the total number of pending tasks.
    pub fn pending_count(&self) -> usize {
        self.deques.iter()
            .map(|d| d.lock().unwrap().len())
            .sum()
    }
}
