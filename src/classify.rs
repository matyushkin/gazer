//! Fixation and saccade classification module.

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
    /// Centroid X (normalized).
    pub x: f64,
    /// Centroid Y (normalized).
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
    /// Peak velocity in degrees per second.
    pub peak_velocity: f64,
}
