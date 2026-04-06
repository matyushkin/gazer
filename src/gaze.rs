//! Gaze estimation module.

/// A 2D gaze point on the screen.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GazePoint {
    /// Normalized X coordinate in [0.0, 1.0] (left to right).
    pub x: f64,
    /// Normalized Y coordinate in [0.0, 1.0] (top to bottom).
    pub y: f64,
    /// Timestamp in milliseconds.
    pub timestamp_ms: u64,
    /// Estimation confidence in [0.0, 1.0].
    pub confidence: f64,
}

/// A 3D gaze vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GazeVector {
    /// Origin point (eye position).
    pub origin: [f64; 3],
    /// Direction unit vector.
    pub direction: [f64; 3],
}
