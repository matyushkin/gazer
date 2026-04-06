//! Blink detection from pupil detection stream.
//!
//! Detects blinks by tracking when pupil detection fails (confidence drops).
//! Also detects prolonged eye closure.

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

/// Current eye state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EyeState {
    /// Eye is open, pupil detected.
    Open,
    /// Eye is closing / in a blink (pupil lost for < closed_threshold).
    Blinking,
    /// Eye has been closed for a prolonged period.
    Closed,
}

/// Stateful blink detector.
///
/// Feed it pupil detection confidence each frame; it tracks blinks
/// and eye state over time.
pub struct BlinkDetector {
    /// Confidence below this = eye not visible. Default: 0.1.
    pub confidence_threshold: f64,
    /// Minimum blink duration in ms to count as a real blink. Default: 50.
    pub min_blink_ms: u64,
    /// Maximum blink duration in ms (beyond this = eyes closed). Default: 500.
    pub closed_threshold_ms: u64,

    state: EyeState,
    /// Timestamp when eye was last seen open.
    last_open_ms: u64,
    /// Timestamp when eye started closing.
    close_start_ms: u64,
    /// Completed blinks.
    blinks: Vec<Blink>,
    /// Total blink count.
    blink_count: u64,
}

impl BlinkDetector {
    pub fn new() -> Self {
        Self {
            confidence_threshold: 0.1,
            min_blink_ms: 50,
            closed_threshold_ms: 500,
            state: EyeState::Open,
            last_open_ms: 0,
            close_start_ms: 0,
            blinks: Vec::new(),
            blink_count: 0,
        }
    }

    /// Update with a new frame's pupil detection confidence and timestamp.
    ///
    /// Returns the current eye state after this update.
    pub fn update(&mut self, confidence: f64, timestamp_ms: u64) -> EyeState {
        let eye_visible = confidence > self.confidence_threshold;

        match self.state {
            EyeState::Open => {
                if eye_visible {
                    self.last_open_ms = timestamp_ms;
                } else {
                    // Eye just disappeared — start potential blink
                    self.close_start_ms = timestamp_ms;
                    self.state = EyeState::Blinking;
                }
            }
            EyeState::Blinking => {
                if eye_visible {
                    // Eye re-appeared — blink completed
                    let duration = timestamp_ms.saturating_sub(self.close_start_ms);
                    if duration >= self.min_blink_ms {
                        let blink = Blink {
                            start_ms: self.close_start_ms,
                            end_ms: timestamp_ms,
                        };
                        self.blinks.push(blink);
                        self.blink_count += 1;
                    }
                    self.state = EyeState::Open;
                    self.last_open_ms = timestamp_ms;
                } else {
                    // Still closed — check if prolonged
                    let duration = timestamp_ms.saturating_sub(self.close_start_ms);
                    if duration > self.closed_threshold_ms {
                        self.state = EyeState::Closed;
                    }
                }
            }
            EyeState::Closed => {
                if eye_visible {
                    // Eyes opened after prolonged closure
                    self.state = EyeState::Open;
                    self.last_open_ms = timestamp_ms;
                }
            }
        }

        self.state
    }

    /// Current eye state.
    pub fn state(&self) -> EyeState {
        self.state
    }

    /// Total number of completed blinks.
    pub fn blink_count(&self) -> u64 {
        self.blink_count
    }

    /// Get the last N blinks.
    pub fn recent_blinks(&self, n: usize) -> &[Blink] {
        let start = self.blinks.len().saturating_sub(n);
        &self.blinks[start..]
    }

    /// Average blinks per minute based on observation window.
    pub fn blinks_per_minute(&self, current_ms: u64, window_ms: u64) -> f64 {
        let cutoff = current_ms.saturating_sub(window_ms);
        let count = self.blinks.iter().filter(|b| b.start_ms >= cutoff).count();
        if window_ms > 0 {
            count as f64 / (window_ms as f64 / 60_000.0)
        } else {
            0.0
        }
    }

    /// Reset the detector state.
    pub fn reset(&mut self) {
        self.state = EyeState::Open;
        self.blinks.clear();
        self.blink_count = 0;
    }
}

impl Default for BlinkDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_normal_blink() {
        let mut det = BlinkDetector::new();

        // Eyes open for 500ms
        for t in (0..500).step_by(33) {
            assert_eq!(det.update(0.5, t), EyeState::Open);
        }

        // Blink: eyes closed for 150ms
        assert_eq!(det.update(0.0, 500), EyeState::Blinking);
        assert_eq!(det.update(0.0, 550), EyeState::Blinking);
        assert_eq!(det.update(0.0, 600), EyeState::Blinking);
        assert_eq!(det.update(0.0, 650), EyeState::Blinking);

        // Eyes open again
        assert_eq!(det.update(0.5, 650), EyeState::Open);
        assert_eq!(det.blink_count(), 1);
        assert_eq!(det.recent_blinks(1)[0].duration_ms(), 150);
    }

    #[test]
    fn detects_prolonged_closure() {
        let mut det = BlinkDetector::new();
        det.update(0.5, 0);

        // Close eyes
        det.update(0.0, 100);
        assert_eq!(det.state(), EyeState::Blinking);

        // Still closed at 600ms — should transition to Closed
        det.update(0.0, 700);
        assert_eq!(det.state(), EyeState::Closed);

        // Open again
        det.update(0.5, 800);
        assert_eq!(det.state(), EyeState::Open);
        // Prolonged closure is NOT counted as a blink
        assert_eq!(det.blink_count(), 0);
    }

    #[test]
    fn ignores_very_short_flicker() {
        let mut det = BlinkDetector::new();
        det.update(0.5, 0);

        // Very short loss (30ms) — below min_blink_ms
        det.update(0.0, 100);
        det.update(0.5, 130); // only 30ms

        assert_eq!(det.blink_count(), 0);
    }

    #[test]
    fn blinks_per_minute_calculation() {
        let mut det = BlinkDetector::new();

        // Simulate 3 blinks over 10 seconds
        for i in 0..3 {
            let base = i * 3000;
            det.update(0.5, base);
            det.update(0.0, base + 1000);
            det.update(0.5, base + 1200);
        }

        let bpm = det.blinks_per_minute(9200, 10_000);
        assert!((bpm - 18.0).abs() < 1.0, "bpm={bpm}"); // 3 blinks in 10s = 18 bpm
    }
}
