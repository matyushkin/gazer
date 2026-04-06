//! Test pupil detection on a static eye image.
//!
//! Usage:
//!   cargo run --release --example test_on_image -- <path_to_eye_image.png>
//!
//! The input should be a cropped eye region (grayscale or RGB).
//! Outputs detection results and saves an annotated image.

use image::{GrayImage, ImageReader, Rgb, RgbImage};
use saccade::ellipse::Ellipse;
use saccade::frame::GrayFrame;
use saccade::pure::{self, PureConfig};
use saccade::timm::{self, TimmConfig};
use saccade::tracker::{Tracker, TrackerConfig};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <eye_image.png> [output.png]", args[0]);
        eprintln!("\nProvide a cropped eye region image (grayscale or RGB).");
        eprintln!("If no image is provided, a synthetic eye will be used for testing.");
        // Use synthetic eye for demo
        run_synthetic();
        return;
    }

    let input_path = &args[1];
    let output_path = if args.len() > 2 {
        args[2].clone()
    } else {
        format!("{}_detected.png", input_path.trim_end_matches(".png").trim_end_matches(".jpg"))
    };

    let img = ImageReader::open(input_path)
        .expect("Failed to open image")
        .decode()
        .expect("Failed to decode image");

    let gray = img.to_luma8();
    let (w, h) = (gray.width(), gray.height());
    println!("Loaded image: {w}×{h} from {input_path}");

    let frame = GrayFrame::new(w, h, gray.as_raw());
    run_detection(&frame, w, h, &gray, &output_path);
}

fn run_synthetic() {
    println!("No image provided. Running on synthetic eye...\n");
    let (w, h) = (160, 120);
    let (cx, cy, rx, ry) = (80.0, 60.0, 25.0, 18.0);

    let mut data = vec![190u8; (w * h) as usize];
    for y in 0..h {
        for x in 0..w {
            let d = (((x as f64 - cx) / rx).powi(2) + ((y as f64 - cy) / ry).powi(2)).sqrt();
            if d < 0.9 {
                data[(y * w + x) as usize] = 30;
            } else if d < 1.1 {
                let t = (d - 0.9) / 0.2;
                data[(y * w + x) as usize] = (30.0 + 160.0 * t) as u8;
            }
        }
    }

    let gray = GrayImage::from_raw(w, h, data.clone()).unwrap();
    let frame = GrayFrame::new(w, h, &data);
    run_detection(&frame, w, h, &gray, "synthetic_detected.png");
}

fn run_detection(frame: &GrayFrame, w: u32, h: u32, gray: &GrayImage, output_path: &str) {
    // 1. Timm & Barth
    let t0 = Instant::now();
    let timm_result = timm::detect_center(frame, &TimmConfig::default());
    let timm_time = t0.elapsed();
    println!("--- Timm & Barth ---");
    println!("  Center: ({:.1}, {:.1})", timm_result.x, timm_result.y);
    println!("  Confidence: {:.3}", timm_result.confidence);
    println!("  Time: {timm_time:.2?}");

    // 2. PuRe
    let t0 = Instant::now();
    let pure_config = PureConfig {
        canny_low: 10.0,
        canny_high: 30.0,
        ..PureConfig::default()
    };
    let pure_result = pure::detect(frame, &pure_config);
    let pure_time = t0.elapsed();
    println!("\n--- PuRe ---");
    if let Some(ref pupil) = pure_result.pupil {
        println!("  Ellipse: center=({:.1}, {:.1}), axes=({:.1}, {:.1}), angle={:.2}°",
            pupil.cx, pupil.cy, pupil.a, pupil.b, pupil.angle.to_degrees());
    } else {
        println!("  No pupil detected");
    }
    println!("  Confidence (ψ): {:.3}", pure_result.confidence);
    println!("  Candidates: {}", pure_result.candidates.len());
    println!("  Time: {pure_time:.2?}");

    // 3. GSAR-PuRe Tracker
    let t0 = Instant::now();
    let mut tracker = Tracker::new(TrackerConfig {
        pure: pure_config.clone(),
        ..TrackerConfig::default()
    });
    let track_result = tracker.track(frame);
    let track_time = t0.elapsed();
    println!("\n--- GSAR-PuRe Tracker ---");
    println!("  Mode: {:?}", track_result.mode);
    if let Some(ref pupil) = track_result.pupil {
        println!("  Ellipse: center=({:.1}, {:.1}), axes=({:.1}, {:.1})",
            pupil.cx, pupil.cy, pupil.a, pupil.b);
    } else {
        println!("  No pupil detected");
    }
    println!("  Confidence: {:.3} (grad={:.3}, edge={:.3}, temporal={:.3})",
        track_result.confidence,
        track_result.confidence_detail.gradient,
        track_result.confidence_detail.edge,
        track_result.confidence_detail.temporal);
    println!("  Time: {track_time:.2?}");

    // Save annotated image
    let mut rgb = RgbImage::from_fn(w, h, |x, y| {
        let g = gray.get_pixel(x, y).0[0];
        Rgb([g, g, g])
    });

    // Draw Timm center as green cross
    draw_cross(&mut rgb, timm_result.x as i32, timm_result.y as i32, Rgb([0, 255, 0]));

    // Draw PuRe ellipse as red
    if let Some(ref pupil) = pure_result.pupil {
        draw_ellipse(&mut rgb, pupil, Rgb([255, 0, 0]));
    }

    // Draw tracker ellipse as cyan
    if let Some(ref pupil) = track_result.pupil {
        draw_ellipse(&mut rgb, pupil, Rgb([0, 255, 255]));
    }

    rgb.save(output_path).expect("Failed to save output");
    println!("\nAnnotated image saved to: {output_path}");
    println!("  Green cross = Timm & Barth center");
    println!("  Red ellipse = PuRe detection");
    println!("  Cyan ellipse = GSAR-PuRe tracker");
}

fn draw_cross(img: &mut RgbImage, cx: i32, cy: i32, color: Rgb<u8>) {
    let (w, h) = (img.width() as i32, img.height() as i32);
    for d in -4..=4 {
        let x = cx + d;
        let y = cy;
        if x >= 0 && x < w && y >= 0 && y < h {
            img.put_pixel(x as u32, y as u32, color);
        }
        let x = cx;
        let y = cy + d;
        if x >= 0 && x < w && y >= 0 && y < h {
            img.put_pixel(x as u32, y as u32, color);
        }
    }
}

fn draw_ellipse(img: &mut RgbImage, ell: &Ellipse, color: Rgb<u8>) {
    let (w, h) = (img.width() as i32, img.height() as i32);
    let n = 200;
    for i in 0..n {
        let t = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
        let x = ell.a * t.cos();
        let y = ell.b * t.sin();
        let cos = ell.angle.cos();
        let sin = ell.angle.sin();
        let px = (ell.cx + x * cos - y * sin).round() as i32;
        let py = (ell.cy + x * sin + y * cos).round() as i32;
        if px >= 0 && px < w && py >= 0 && py < h {
            img.put_pixel(px as u32, py as u32, color);
        }
    }
}
