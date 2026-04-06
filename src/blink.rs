//! Blink detection module.

/// A detected blink event.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Blink {
    /// Start timestamp in milliseconds.
    pub start_ms: u64,
    /// End timestamp in milliseconds.
    pub end_ms: u64,
}

impl Blink {
    /// Duration of the blink in milliseconds.
    pub fn duration_ms(&self) -> u64 {
        self.end_ms - self.start_ms
    }
}
