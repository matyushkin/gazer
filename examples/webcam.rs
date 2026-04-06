//! Real-time eye tracking from webcam.
//!
//! Usage:
//!   cargo run --release --features demo --example webcam
//!
//! Requires: webcam access permission on macOS.
//! Downloads face detection model on first run.

use minifb::{Key, Window, WindowOptions};
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{CameraIndex, RequestedFormat, RequestedFormatType};
use nokhwa::Camera;
use rustface::ImageData;
use saccade::frame::GrayFrame;
use saccade::pure::PureConfig;
use saccade::tracker::{Tracker, TrackerConfig, TrackingMode};
use std::time::Instant;

const MODEL_PATH: &str = "seeta_fd_frontal_v1.0.bin";
const MODEL_URL: &str = "https://github.com/atomashpolskiy/rustface/raw/master/model/seeta_fd_frontal_v1.0.bin";

fn main() {
    // Ensure face detection model exists
    if !std::path::Path::new(MODEL_PATH).exists() {
        println!("Downloading face detection model...");
        let status = std::process::Command::new("curl")
            .args(["-L", "-o", MODEL_PATH, MODEL_URL])
            .status()
            .expect("Failed to run curl");
        if !status.success() {
            eprintln!("Failed to download model. Please download manually:");
            eprintln!("  curl -L -o {MODEL_PATH} {MODEL_URL}");
            std::process::exit(1);
        }
        println!("Model downloaded.");
    }

    // Initialize face detector
    let mut detector = rustface::create_detector(MODEL_PATH)
        .expect("Failed to create face detector");
    detector.set_min_face_size(60);
    detector.set_score_thresh(2.0);

    // Initialize camera
    let format = RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);
    let mut camera = Camera::new(CameraIndex::Index(0), format)
        .expect("Failed to open camera");
    camera.open_stream().expect("Failed to open camera stream");

    let cam_res = camera.resolution();
    println!("Camera native: {}×{}", cam_res.width(), cam_res.height());

    // We'll downscale to ~640px wide for processing speed
    let scale_down = (cam_res.width() / 640).max(1) as usize;
    let cam_w = cam_res.width() as usize / scale_down;
    let cam_h = cam_res.height() as usize / scale_down;
    println!("Processing at: {cam_w}×{cam_h} (scale 1/{scale_down})");

    // Initialize display window
    let win_w = cam_w;
    let win_h = cam_h;
    let mut window = Window::new(
        "Saccade — Eye Tracker (ESC to quit)",
        win_w,
        win_h,
        WindowOptions::default(),
    )
    .expect("Failed to create window");

    window.set_target_fps(60);

    // Initialize eye trackers (left and right)
    let tracker_config = TrackerConfig {
        pure: PureConfig {
            canny_low: 10.0,
            canny_high: 30.0,
            ..PureConfig::default()
        },
        ..TrackerConfig::default()
    };
    let mut left_tracker = Tracker::new(tracker_config.clone());
    let mut right_tracker = Tracker::new(tracker_config);

    let mut frame_buf = vec![0u32; win_w * win_h];
    let mut fps_counter = FpsCounter::new();

    println!("Running... Press ESC to quit.");

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let frame_start = Instant::now();

        // Capture frame — decode to RGB
        let decoded = match camera.frame() {
            Ok(f) => f.decode_image::<RgbFormat>(),
            Err(e) => {
                eprintln!("Frame error: {e}");
                continue;
            }
        };
        let rgb_image = match decoded {
            Ok(img) => img,
            Err(e) => {
                eprintln!("Decode error: {e}");
                continue;
            }
        };
        let full_w = rgb_image.width() as usize;
        let full_h = rgb_image.height() as usize;
        let rgb_full = rgb_image.as_raw();

        // Downscale RGB and grayscale
        let mut rgb_data = vec![0u8; cam_w * cam_h * 3];
        let mut gray = vec![0u8; cam_w * cam_h];
        for y in 0..cam_h {
            for x in 0..cam_w {
                let sx = x * scale_down;
                let sy = y * scale_down;
                if sx < full_w && sy < full_h {
                    let si = (sy * full_w + sx) * 3;
                    let di = (y * cam_w + x) * 3;
                    let (r, g, b) = (rgb_full[si], rgb_full[si + 1], rgb_full[si + 2]);
                    rgb_data[di] = r;
                    rgb_data[di + 1] = g;
                    rgb_data[di + 2] = b;
                    gray[y * cam_w + x] =
                        (0.299 * r as f32 + 0.587 * g as f32 + 0.114 * b as f32) as u8;
                }
            }
        }

        // Detect faces
        let image_data = ImageData::new(&gray, cam_w as u32, cam_h as u32);
        let faces = detector.detect(&image_data);

        // Convert RGB to display buffer (0x00RRGGBB)
        for i in 0..cam_w * cam_h {
            let r = rgb_data[i * 3] as u32;
            let g = rgb_data[i * 3 + 1] as u32;
            let b = rgb_data[i * 3 + 2] as u32;
            frame_buf[i] = (r << 16) | (g << 8) | b;
        }

        // Process each detected face
        for face in &faces {
            let bbox = face.bbox();
            let fx = bbox.x().max(0) as usize;
            let fy = bbox.y().max(0) as usize;
            let fw_face = bbox.width() as usize;
            let fh_face = bbox.height() as usize;

            // Draw face bbox (yellow)
            draw_rect(&mut frame_buf, win_w, win_h, fx, fy, fw_face, fh_face, 0xFFFF00);

            // Estimate eye regions from face bbox
            // Eyes are roughly in upper 35-55% of face, left/right halves
            let eye_y = fy + fh_face * 30 / 100;
            let eye_h = fh_face * 25 / 100;
            let left_eye_x = fx + fw_face * 5 / 100;
            let right_eye_x = fx + fw_face * 52 / 100;
            let eye_w = fw_face * 43 / 100;

            // Extract and track left eye
            if let Some(result) = track_eye(
                &gray, cam_w, cam_h,
                left_eye_x, eye_y, eye_w, eye_h,
                &mut left_tracker,
            ) {
                draw_rect(&mut frame_buf, win_w, win_h, left_eye_x, eye_y, eye_w, eye_h, 0x00FF00);
                if let Some(pupil) = result.pupil {
                    let px = (left_eye_x as f64 + pupil.cx) as usize;
                    let py = (eye_y as f64 + pupil.cy) as usize;
                    draw_crosshair(&mut frame_buf, win_w, win_h, px, py, 0xFF0000, result.mode);
                }
            }

            // Extract and track right eye
            if let Some(result) = track_eye(
                &gray, cam_w, cam_h,
                right_eye_x, eye_y, eye_w, eye_h,
                &mut right_tracker,
            ) {
                draw_rect(&mut frame_buf, win_w, win_h, right_eye_x, eye_y, eye_w, eye_h, 0x00FF00);
                if let Some(pupil) = result.pupil {
                    let px = (right_eye_x as f64 + pupil.cx) as usize;
                    let py = (eye_y as f64 + pupil.cy) as usize;
                    draw_crosshair(&mut frame_buf, win_w, win_h, px, py, 0x00FFFF, result.mode);
                }
            }
        }

        // Draw FPS
        fps_counter.tick();
        let fps = fps_counter.fps();
        let frame_ms = frame_start.elapsed().as_millis();
        // FPS indicator: top-left colored bar
        let bar_len = (fps as usize).min(win_w);
        let bar_color = if fps > 20.0 { 0x00FF00 } else if fps > 10.0 { 0xFFFF00 } else { 0xFF0000 };
        for x in 0..bar_len {
            for y in 0..4 {
                frame_buf[y * win_w + x] = bar_color;
            }
        }

        if faces.is_empty() {
            // No face detected — reset trackers
            left_tracker.reset();
            right_tracker.reset();
        }

        window.update_with_buffer(&frame_buf, win_w, win_h).unwrap();

        if fps_counter.frame_count % 30 == 0 {
            print!("\rFPS: {fps:.1} | Frame: {frame_ms}ms | Faces: {}    ", faces.len());
        }
    }

    println!("\nDone.");
}

struct TrackResult {
    pupil: Option<saccade::ellipse::Ellipse>,
    mode: TrackingMode,
}

fn track_eye(
    gray: &[u8],
    img_w: usize,
    img_h: usize,
    roi_x: usize,
    roi_y: usize,
    roi_w: usize,
    roi_h: usize,
    tracker: &mut Tracker,
) -> Option<TrackResult> {
    if roi_x + roi_w > img_w || roi_y + roi_h > img_h || roi_w < 20 || roi_h < 15 {
        return None;
    }

    // Extract eye region
    let mut eye_data = vec![0u8; roi_w * roi_h];
    for y in 0..roi_h {
        let src_start = (roi_y + y) * img_w + roi_x;
        let dst_start = y * roi_w;
        eye_data[dst_start..dst_start + roi_w]
            .copy_from_slice(&gray[src_start..src_start + roi_w]);
    }

    let frame = GrayFrame::new(roi_w as u32, roi_h as u32, &eye_data);
    let result = tracker.track(&frame);

    Some(TrackResult {
        pupil: result.pupil,
        mode: result.mode,
    })
}

fn draw_rect(buf: &mut [u32], w: usize, h: usize, x: usize, y: usize, rw: usize, rh: usize, color: u32) {
    for dx in 0..rw {
        let px = x + dx;
        if px < w {
            if y < h { buf[y * w + px] = color; }
            if y + rh < h { buf[(y + rh) * w + px] = color; }
        }
    }
    for dy in 0..rh {
        let py = y + dy;
        if py < h {
            if x < w { buf[py * w + x] = color; }
            if x + rw < w { buf[py * w + x + rw - 1] = color; }
        }
    }
}

fn draw_crosshair(buf: &mut [u32], w: usize, h: usize, cx: usize, cy: usize, color: u32, mode: TrackingMode) {
    let size = match mode {
        TrackingMode::Fast => 4,
        TrackingMode::Precise => 6,
        TrackingMode::FullScan => 8,
    };
    for d in -(size as i32)..=size as i32 {
        let x = cx as i32 + d;
        let y = cy as i32;
        if x >= 0 && (x as usize) < w && (y as usize) < h {
            buf[y as usize * w + x as usize] = color;
        }
        let x = cx as i32;
        let y = cy as i32 + d;
        if x >= 0 && (x as usize) < w && y >= 0 && (y as usize) < h {
            buf[y as usize * w + x as usize] = color;
        }
    }
    // Circle around crosshair
    for i in 0..32 {
        let t = 2.0 * std::f64::consts::PI * i as f64 / 32.0;
        let r = (size + 2) as f64;
        let x = (cx as f64 + r * t.cos()).round() as i32;
        let y = (cy as f64 + r * t.sin()).round() as i32;
        if x >= 0 && (x as usize) < w && y >= 0 && (y as usize) < h {
            buf[y as usize * w + x as usize] = color;
        }
    }
}

struct FpsCounter {
    last_time: Instant,
    frame_count: u64,
    fps: f64,
}

impl FpsCounter {
    fn new() -> Self {
        Self {
            last_time: Instant::now(),
            frame_count: 0,
            fps: 0.0,
        }
    }

    fn tick(&mut self) {
        self.frame_count += 1;
        let elapsed = self.last_time.elapsed().as_secs_f64();
        if elapsed >= 1.0 {
            self.fps = self.frame_count as f64 / elapsed;
            self.frame_count = 0;
            self.last_time = Instant::now();
        }
    }

    fn fps(&self) -> f64 {
        self.fps
    }
}
