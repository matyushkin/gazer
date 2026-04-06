//! Pupil detection module.

/// Detected pupil with position and radius.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pupil {
    /// X coordinate in image pixels.
    pub x: f64,
    /// Y coordinate in image pixels.
    pub y: f64,
    /// Estimated radius in pixels.
    pub radius: f64,
    /// Detection confidence in [0.0, 1.0].
    pub confidence: f64,
}

/// Detection result for both eyes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PupilPair {
    pub left: Option<Pupil>,
    pub right: Option<Pupil>,
}
