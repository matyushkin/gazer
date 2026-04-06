//! Fixation and saccade classification using I-VT (velocity threshold) algorithm.
//!
//! Classifies a stream of gaze/pupil positions into:
//! - **Fixations**: gaze held steady on a region
//! - **Saccades**: rapid eye movements between fixation points

use crate::gaze::GazePoint;

/// Classified eye movement event.
#[derive(Debug, Clone, PartialEq)]
pub enum EyeEvent {
    /// Gaze held relatively steady on a point.
    Fixation(Fixation),
    /// Rapid movement between fixation points.
    Saccade(Saccade),
}

/// A fixation: gaze resting on a region.
#[derive(Debug, Clone, PartialEq)]
pub struct Fixation {
    /// Centroid X (normalized or pixel coordinates).
    pub x: f64,
    /// Centroid Y.
    pub y: f64,
    /// Start timestamp in milliseconds.
    pub start_ms: u64,
    /// Duration in milliseconds.
    pub duration_ms: u64,
}

/// A saccade: rapid eye movement.
#[derive(Debug, Clone, PartialEq)]
pub struct Saccade {
    /// Start point.
    pub from: GazePoint,
    /// End point.
    pub to: GazePoint,
    /// Peak velocity in pixels per second.
    pub peak_velocity: f64,
}

/// I-VT (Velocity-Threshold Identification) classifier.
///
/// Points with velocity below the threshold are classified as fixation;
/// above are saccade.
pub struct IVTClassifier {
    /// Velocity threshold in pixels per second. Default: 50.0.
    pub velocity_threshold: f64,
    /// Minimum fixation duration in ms. Default: 100.
    pub min_fixation_ms: u64,

    // State
    prev_point: Option<(f64, f64, u64)>, // (x, y, timestamp_ms)
    fixation_points: Vec<(f64, f64, u64)>,
    events: Vec<EyeEvent>,
    saccade_start: Option<GazePoint>,
    peak_vel: f64,
}

impl IVTClassifier {
    pub fn new(velocity_threshold: f64, min_fixation_ms: u64) -> Self {
        Self {
            velocity_threshold,
            min_fixation_ms,
            prev_point: None,
            fixation_points: Vec::new(),
            events: Vec::new(),
            saccade_start: None,
            peak_vel: 0.0,
        }
    }

    /// Create with default parameters.
    pub fn default_params() -> Self {
        Self::new(50.0, 100)
    }

    /// Feed a new gaze point. Returns any newly completed events.
    pub fn update(&mut self, x: f64, y: f64, timestamp_ms: u64) -> Vec<EyeEvent> {
        let mut new_events = Vec::new();

        if let Some((px, py, pt)) = self.prev_point {
            let dt = (timestamp_ms - pt) as f64 / 1000.0; // seconds
            if dt > 0.0 {
                let dx = x - px;
                let dy = y - py;
                let velocity = (dx * dx + dy * dy).sqrt() / dt; // pixels/sec

                if velocity < self.velocity_threshold {
                    // Fixation point
                    // If we were in a saccade, end it
                    if let Some(from) = self.saccade_start.take() {
                        let to = GazePoint {
                            x, y,
                            timestamp_ms,
                            confidence: 1.0,
                        };
                        new_events.push(EyeEvent::Saccade(Saccade {
                            from,
                            to,
                            peak_velocity: self.peak_vel,
                        }));
                        self.peak_vel = 0.0;
                    }
                    self.fixation_points.push((x, y, timestamp_ms));
                } else {
                    // Saccade point
                    // End any ongoing fixation
                    if let Some(event) = self.end_fixation() {
                        new_events.push(event);
                    }
                    if self.saccade_start.is_none() {
                        self.saccade_start = Some(GazePoint {
                            x: px, y: py,
                            timestamp_ms: pt,
                            confidence: 1.0,
                        });
                    }
                    self.peak_vel = self.peak_vel.max(velocity);
                }
            }
        }

        self.prev_point = Some((x, y, timestamp_ms));
        self.events.extend(new_events.iter().cloned());
        new_events
    }

    /// End any ongoing fixation and return the event.
    fn end_fixation(&mut self) -> Option<EyeEvent> {
        if self.fixation_points.len() < 2 {
            self.fixation_points.clear();
            return None;
        }

        let start_ms = self.fixation_points.first().unwrap().2;
        let end_ms = self.fixation_points.last().unwrap().2;
        let duration = end_ms - start_ms;

        if duration < self.min_fixation_ms {
            self.fixation_points.clear();
            return None;
        }

        let n = self.fixation_points.len() as f64;
        let cx: f64 = self.fixation_points.iter().map(|p| p.0).sum::<f64>() / n;
        let cy: f64 = self.fixation_points.iter().map(|p| p.1).sum::<f64>() / n;

        self.fixation_points.clear();

        Some(EyeEvent::Fixation(Fixation {
            x: cx,
            y: cy,
            start_ms,
            duration_ms: duration,
        }))
    }

    /// Flush any pending fixation (call when stream ends).
    pub fn flush(&mut self) -> Vec<EyeEvent> {
        let mut events = Vec::new();
        if let Some(event) = self.end_fixation() {
            events.push(event);
        }
        events
    }

    /// All events collected so far.
    pub fn events(&self) -> &[EyeEvent] {
        &self.events
    }

    /// Reset state.
    pub fn reset(&mut self) {
        self.prev_point = None;
        self.fixation_points.clear();
        self.events.clear();
        self.saccade_start = None;
        self.peak_vel = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_fixation() {
        let mut clf = IVTClassifier::new(50.0, 100);

        // Simulate steady gaze at (100, 100) for 500ms at 30fps
        for i in 0..15 {
            let t = i * 33;
            // Small jitter (< threshold)
            let x = 100.0 + (i % 3) as f64 * 0.5;
            clf.update(x, 100.0, t);
        }

        let events = clf.flush();
        assert_eq!(events.len(), 1);
        match &events[0] {
            EyeEvent::Fixation(f) => {
                assert!((f.x - 100.5).abs() < 1.0);
                assert!(f.duration_ms >= 100);
            }
            _ => panic!("expected fixation"),
        }
    }

    #[test]
    fn detects_saccade_between_fixations() {
        let mut clf = IVTClassifier::new(50.0, 100);

        // Fixation at (100, 100)
        for i in 0..10 {
            clf.update(100.0, 100.0, i * 33);
        }

        // Rapid jump to (300, 100) — saccade
        clf.update(300.0, 100.0, 330 + 16);

        // Fixation at (300, 100)
        for i in 0..10 {
            clf.update(300.0, 100.0, 350 + i * 33);
        }

        clf.flush();

        // Check all accumulated events
        let all = clf.events();
        let fixations: Vec<_> = all.iter().filter(|e| matches!(e, EyeEvent::Fixation(_))).collect();
        let saccades: Vec<_> = all.iter().filter(|e| matches!(e, EyeEvent::Saccade(_))).collect();

        assert!(fixations.len() >= 1, "should have at least 1 fixation, got {}", fixations.len());
        assert_eq!(saccades.len(), 1, "should have 1 saccade");

        if let EyeEvent::Saccade(s) = &saccades[0] {
            assert!(s.peak_velocity > 1000.0, "saccade should be fast: {}", s.peak_velocity);
        }
    }

    #[test]
    fn no_events_for_short_fixation() {
        let mut clf = IVTClassifier::new(50.0, 100);
        // Only 2 frames — too short for fixation
        clf.update(100.0, 100.0, 0);
        clf.update(100.0, 100.0, 33);
        let events = clf.flush();
        assert!(events.is_empty());
    }
}
