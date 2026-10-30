use std::hash::{Hash, Hasher};

/// Route requests to different model versions based on user ID.
fn route_to_model<'a>(user_id: &str, version_a: &'a str, version_b: &'a str) -> &'a str {
    // Simple hash-based routing: ~50% to each version.
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    user_id.hash(&mut hasher);
    let hash = hasher.finish();

    if hash % 2 == 0 {
        version_a
    } else {
        version_b
    }
}
