//! CNN-based webcam eye tracker using MobileGaze (pretrained gaze direction CNN).
//!
//! Pipeline:
//!   camera → rustface (face bbox) → MobileGaze CNN → (yaw, pitch) angles
//!   → calibration: linear (yaw,pitch) → (screen_x, screen_y) mapping (6 params)
//!   → smoothing → cursor
//!
//! Why CNN: MobileGaze is trained on thousands of faces with ground-truth gaze.
//! It understands gaze direction in absolute terms, not pixel positions.
//! Should be more robust to head movement than pixel-feature ridge regression.
//!
//! Calibration is much simpler: just 6 parameters to fit, so 5-9 points is enough.

use display_info::DisplayInfo;
use minifb::{Key, MouseButton, MouseMode, Window, WindowOptions};
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{CameraIndex, RequestedFormat, RequestedFormatType};
use nokhwa::Camera;
use rustface::ImageData;
use saccade::calib_state::{CalibrationState, EventResult, Phase};
use saccade::one_euro::OneEuroFilter2D;
use std::time::Instant;
use tract_onnx::prelude::*;

type Model = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

const FACE_M: &str = "seeta_fd_frontal_v1.0.bin";
const FACE_U: &str = "https://github.com/atomashpolskiy/rustface/raw/master/model/seeta_fd_frontal_v1.0.bin";
const GAZE_M: &str = "mobileone_s0_gaze.onnx";
const GAZE_U: &str = "https://github.com/yakhyo/gaze-estimation/releases/download/weights/mobileone_s0_gaze.onnx";

fn dl(p: &str, u: &str) {
    if !std::path::Path::new(p).exists() {
        println!("Downloading {p}...");
        let _ = std::process::Command::new("curl").args(["-L", "-o", p, u]).status();
    }
}
fn load(p: &str) -> Model {
    tract_onnx::onnx().model_for_path(p).unwrap().into_optimized().unwrap().into_runnable().unwrap()
}

fn calib_points(w: usize, h: usize) -> Vec<(f64, f64)> {
    let mx = w as f64 * 0.1;
    let my = h as f64 * 0.1;
    let cx = w as f64 / 2.0;
    let cy = h as f64 / 2.0;
    vec![
        (mx, my), (cx, my), (w as f64 - mx, my),
        (mx, cy), (w as f64 - mx, cy),
        (mx, h as f64 - my), (cx, h as f64 - my), (w as f64 - mx, h as f64 - my),
        (cx, cy),
    ]
}

/// Linear regression: (yaw, pitch) → (screen_x, screen_y).
/// 6 parameters total: ax, bx, cx, ay, by, cy where
/// screen_x = ax + bx*yaw + cx*pitch
/// screen_y = ay + by*yaw + cy*pitch
struct LinearMapper {
    samples: Vec<((f32, f32), (f32, f32))>, // ((yaw, pitch), (screen_x, screen_y))
    coeffs_x: [f64; 3], // [a, b_yaw, c_pitch]
    coeffs_y: [f64; 3],
    fitted: bool,
}

impl LinearMapper {
    fn new() -> Self {
        Self {
            samples: Vec::new(),
            coeffs_x: [0.0; 3],
            coeffs_y: [0.0; 3],
            fitted: false,
        }
    }

    fn add(&mut self, yaw: f32, pitch: f32, sx: f32, sy: f32) {
        self.samples.push(((yaw, pitch), (sx, sy)));
    }

    fn clear(&mut self) {
        self.samples.clear();
        self.fitted = false;
    }

    /// Fit via normal equations: solve [1, yaw, pitch] @ β = screen.
    fn fit(&mut self) -> bool {
        let n = self.samples.len();
        if n < 3 { return false; }

        // Design matrix A (n × 3) and targets bx, by
        let mut a = vec![0.0f64; n * 3];
        let mut bx = vec![0.0f64; n];
        let mut by = vec![0.0f64; n];
        for (i, ((yaw, pitch), (sx, sy))) in self.samples.iter().enumerate() {
            a[i*3]   = 1.0;
            a[i*3+1] = *yaw as f64;
            a[i*3+2] = *pitch as f64;
            bx[i] = *sx as f64;
            by[i] = *sy as f64;
        }

        // AᵀA (3×3)
        let mut ata = [[0.0f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                let mut s = 0.0;
                for k in 0..n { s += a[k*3+i] * a[k*3+j]; }
                ata[i][j] = s;
            }
        }
        // Add tiny ridge for stability
        for i in 0..3 { ata[i][i] += 1e-6; }

        let mut atbx = [0.0f64; 3];
        let mut atby = [0.0f64; 3];
        for i in 0..3 {
            for k in 0..n {
                atbx[i] += a[k*3+i] * bx[k];
                atby[i] += a[k*3+i] * by[k];
            }
        }

        // Solve 3×3 system via Cramer or Gaussian — use Cramer
        let det = det3(&ata);
        if det.abs() < 1e-12 { return false; }
        for i in 0..3 {
            let mut a_x = ata; let mut a_y = ata;
            for r in 0..3 { a_x[r][i] = atbx[r]; a_y[r][i] = atby[r]; }
            self.coeffs_x[i] = det3(&a_x) / det;
            self.coeffs_y[i] = det3(&a_y) / det;
        }
        self.fitted = true;
        true
    }

    fn predict(&self, yaw: f32, pitch: f32) -> Option<(f32, f32)> {
        if !self.fitted { return None; }
        let sx = self.coeffs_x[0] + self.coeffs_x[1] * yaw as f64 + self.coeffs_x[2] * pitch as f64;
        let sy = self.coeffs_y[0] + self.coeffs_y[1] * yaw as f64 + self.coeffs_y[2] * pitch as f64;
        Some((sx as f32, sy as f32))
    }

    fn loo_error(&self) -> f64 {
        let n = self.samples.len();
        if n < 4 { return f64::INFINITY; }
        let mut errors = Vec::new();
        for i in 0..n {
            let mut tmp = LinearMapper::new();
            for (j, s) in self.samples.iter().enumerate() {
                if j != i { tmp.add(s.0.0, s.0.1, s.1.0, s.1.1); }
            }
            if !tmp.fit() { continue; }
            let held = &self.samples[i];
            if let Some((px, py)) = tmp.predict(held.0.0, held.0.1) {
                let dx = px as f64 - held.1.0 as f64;
                let dy = py as f64 - held.1.1 as f64;
                errors.push((dx*dx + dy*dy).sqrt());
            }
        }
        if errors.is_empty() { f64::INFINITY }
        else { errors.iter().sum::<f64>() / errors.len() as f64 }
    }
}

fn det3(m: &[[f64;3];3]) -> f64 {
    m[0][0]*(m[1][1]*m[2][2] - m[1][2]*m[2][1])
    - m[0][1]*(m[1][0]*m[2][2] - m[1][2]*m[2][0])
    + m[0][2]*(m[1][0]*m[2][1] - m[1][1]*m[2][0])
}

fn main() {
    dl(FACE_M, FACE_U);
    dl(GAZE_M, GAZE_U);

    let mut face_det = rustface::create_detector(FACE_M).expect("face model");
    face_det.set_min_face_size(80);
    face_det.set_score_thresh(2.0);
    let gaze_net = load(GAZE_M);

    let format = RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);
    let mut camera = loop {
        match Camera::new(CameraIndex::Index(0), format.clone()) {
            Ok(c) => break c,
            Err(e) => { eprintln!("Camera: {e}. Retry..."); std::thread::sleep(std::time::Duration::from_secs(2)); }
        }
    };
    camera.open_stream().expect("stream");
    let cam_res = camera.resolution();
    let sd = (cam_res.width() / 720).max(1) as usize;
    let cw = cam_res.width() as usize / sd;
    let ch = cam_res.height() as usize / sd;
    println!("Camera: {}x{} -> {}x{}", cam_res.width(), cam_res.height(), cw, ch);

    let displays = DisplayInfo::all().expect("displays");
    let primary = displays.iter().find(|d| d.is_primary).unwrap_or(&displays[0]);
    let sw = primary.width as usize;
    let sh = primary.height as usize;
    println!("Screen: {sw}x{sh}");

    let mut window = Window::new(
        "Saccade CNN [click dots, ESC quit]",
        sw, sh,
        WindowOptions { ..WindowOptions::default() },
    ).unwrap();
    window.set_position(0, 0);
    window.set_target_fps(60);

    let mut mapper = LinearMapper::new();
    let mut one_euro = OneEuroFilter2D::new(1.0, 0.05, 1.0);

    let targets = calib_points(sw, sh);
    // Only 1 click per point — linear mapping is simple, doesn't need many samples
    const SAMPLES_PER_POINT: u32 = 2;
    let mut calib = CalibrationState::new(targets.len(), SAMPLES_PER_POINT);

    let validation_targets: Vec<(f64, f64)> = vec![
        (sw as f64 * 0.2, sh as f64 * 0.2),
        (sw as f64 * 0.8, sh as f64 * 0.2),
        (sw as f64 / 2.0, sh as f64 / 2.0),
        (sw as f64 * 0.2, sh as f64 * 0.8),
        (sw as f64 * 0.8, sh as f64 * 0.8),
    ];
    let mut validation_idx = 0usize;
    let mut validation_results: Vec<(f64, f64, f64, f64)> = Vec::new();
    let mut prev_val_mouse_down = false;

    let mut buf = vec![0u32; sw * sh];
    let mut face_sm = SmRect::new(0.3);
    let mut frame_n = 0u64;

    let pip_w = 240usize;
    let pip_h = pip_w * ch / cw;
    let start = Instant::now();
    let mut fps_c = FpsC::new();
    let mut prev_mouse_down = false;
    let mut auto_started = false;

    println!("Click each red dot {SAMPLES_PER_POINT}× to calibrate (only {} clicks total).",
        targets.len() * SAMPLES_PER_POINT as usize);

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let now_ms = start.elapsed().as_millis() as u64;
        let mouse_down = window.get_mouse_down(MouseButton::Left);
        let mouse_edge = mouse_down && !prev_mouse_down;
        prev_mouse_down = mouse_down;
        let mouse_pos = window.get_mouse_pos(MouseMode::Discard).unwrap_or((0.0, 0.0));

        if window.is_key_down(Key::C) {
            calib.restart();
            mapper.clear();
            one_euro.reset();
            println!("\nRecalibrating.");
        }

        if !auto_started && calib.phase() == Phase::Idle {
            calib.start();
            mapper.clear();
            auto_started = true;
            println!("Calibration auto-started.");
        }

        let decoded = match camera.frame() {
            Ok(f) => match f.decode_image::<RgbFormat>() { Ok(i) => i, Err(_) => continue },
            Err(_) => continue,
        };
        let (fw, fh) = (decoded.width() as usize, decoded.height() as usize);
        let rf = decoded.as_raw();
        let mut rgb = vec![0u8; cw*ch*3];
        let mut gray = vec![0u8; cw*ch];
        for y in 0..ch { for x in 0..cw {
            let (sx, sy) = (x*sd, y*sd);
            if sx<fw && sy<fh {
                let si=(sy*fw+sx)*3; let di=(y*cw+x)*3;
                rgb[di]=rf[si]; rgb[di+1]=rf[si+1]; rgb[di+2]=rf[si+2];
                gray[y*cw+x]=(0.299*rf[si] as f32+0.587*rf[si+1] as f32+0.114*rf[si+2] as f32) as u8;
            }
        }}

        frame_n += 1;
        let do_face = frame_n % 3 == 1 || !face_sm.init;
        let faces = if do_face {
            face_det.detect(&ImageData::new(&gray, cw as u32, ch as u32))
        } else { Vec::new() };
        let detected = faces.iter().max_by_key(|f| { let b=f.bbox(); b.width()*b.height() });

        for px in buf.iter_mut() { *px = 0; }

        // PIP preview
        let pip_x = sw - pip_w - 10;
        let pip_y = 10;
        for py in 0..pip_h { for px in 0..pip_w {
            let sx2 = px * cw / pip_w;
            let sy2 = py * ch / pip_h;
            if sx2 < cw && sy2 < ch {
                let si = (sy2 * cw + sx2) * 3;
                buf[(pip_y + py) * sw + pip_x + px] = ((rgb[si] as u32)<<16)|((rgb[si+1] as u32)<<8)|rgb[si+2] as u32;
            }
        }}
        draw_rect(&mut buf, sw, sh, pip_x.saturating_sub(1), pip_y.saturating_sub(1), pip_w+2, pip_h+2, 0xFFFFFF);

        let mut current_gaze: Option<(f32, f32)> = None;
        if let Some(face) = detected {
            let bb = face.bbox();
            face_sm.update(bb.x().max(0) as f64, bb.y().max(0) as f64, bb.width() as f64, bb.height() as f64);
        }

        if face_sm.init {
            let (fx, fy, fwf, fhf) = face_sm.get();
            // Run MobileGaze on face crop
            if let Some((yaw, pitch)) = run_gaze(&gaze_net, &rgb, cw, ch, fx, fy, fwf, fhf) {
                current_gaze = Some((yaw, pitch));
            }
        }

        // Calibration phase
        if calib.phase() == Phase::Calibrating {
            let calib_idx = calib.current_point();
            let samples_at = calib.samples_at_current();
            let (tx, ty) = targets[calib_idx];

            let progress = samples_at as f32 / SAMPLES_PER_POINT as f32;
            let red = (180.0 + 75.0 * progress) as u32;
            let green = (140.0 * (1.0 - progress)) as u32;
            let blue = (140.0 * (1.0 - progress)) as u32;
            let color = (red << 16) | (green << 8) | blue;
            let r = 22 + (progress * 8.0) as usize;
            draw_filled_circle(&mut buf, sw, sh, tx as usize, ty as usize, r, color);
            draw_ring(&mut buf, sw, sh, tx as usize, ty as usize, r + 3, 0xFFFFFF);

            let click_dist = ((mouse_pos.0 as f64 - tx).powi(2) + (mouse_pos.1 as f64 - ty).powi(2)).sqrt();
            if mouse_edge && click_dist < 80.0 {
                let has = current_gaze.is_some();
                let result = calib.handle_capture(has);
                match result {
                    EventResult::SampleCaptured { point, sample } => {
                        if let Some((yaw, pitch)) = current_gaze {
                            mapper.add(yaw, pitch, tx as f32, ty as f32);
                        }
                        println!("  Point {}/{} sample {}/{}", point + 1, targets.len(), sample, SAMPLES_PER_POINT);
                    }
                    EventResult::NextPoint { point } => {
                        if let Some((yaw, pitch)) = current_gaze {
                            mapper.add(yaw, pitch, tx as f32, ty as f32);
                        }
                        println!("  Point {} done, next: {}/{}", calib_idx + 1, point + 1, targets.len());
                    }
                    EventResult::CalibrationComplete => {
                        if let Some((yaw, pitch)) = current_gaze {
                            mapper.add(yaw, pitch, tx as f32, ty as f32);
                        }
                        if mapper.fit() {
                            let loo = mapper.loo_error();
                            println!("\nCalibration done: {} samples. LOO error: {loo:.0} px",
                                mapper.samples.len());
                        }
                        validation_idx = 0;
                        validation_results.clear();
                        one_euro.reset();
                    }
                    _ => {}
                }
            }
        }

        // Validation phase
        if calib.phase() == Phase::Validating && validation_idx < validation_targets.len() {
            let (vtx, vty) = validation_targets[validation_idx];
            draw_filled_circle(&mut buf, sw, sh, vtx as usize, vty as usize, 24, 0x00AAFF);
            draw_ring(&mut buf, sw, sh, vtx as usize, vty as usize, 30, 0xFFFFFF);
            for (i, &(px, py)) in validation_targets.iter().enumerate() {
                let c = if i < validation_idx { 0x00FF00 }
                        else if i == validation_idx { 0x00AAFF }
                        else { 0x333333 };
                draw_filled_circle(&mut buf, sw, sh, px as usize, py as usize, 6, c);
            }

            let val_md = window.get_mouse_down(MouseButton::Left);
            let val_edge = val_md && !prev_val_mouse_down;
            prev_val_mouse_down = val_md;

            let mp = window.get_mouse_pos(MouseMode::Discard).unwrap_or((0.0, 0.0));
            let click_dist = ((mp.0 as f64 - vtx).powi(2) + (mp.1 as f64 - vty).powi(2)).sqrt();
            if val_edge && click_dist < 80.0 {
                if let Some((yaw, pitch)) = current_gaze {
                    if let Some((px, py)) = mapper.predict(yaw, pitch) {
                        let px = (px as f64).clamp(0.0, sw as f64 - 1.0);
                        let py = (py as f64).clamp(0.0, sh as f64 - 1.0);
                        validation_results.push((px, py, vtx, vty));
                        let err = ((px - vtx).powi(2) + (py - vty).powi(2)).sqrt();
                        println!("  Val {}/{}: pred ({px:.0}, {py:.0}), tgt ({vtx:.0}, {vty:.0}), err {err:.0}",
                            validation_idx + 1, validation_targets.len());
                    }
                }
                validation_idx += 1;
            }

            if validation_idx >= validation_targets.len() {
                let errors: Vec<f64> = validation_results.iter()
                    .map(|(px,py,tx,ty)| ((px-tx).powi(2)+(py-ty).powi(2)).sqrt()).collect();
                if !errors.is_empty() {
                    let n = errors.len() as f64;
                    let mean = errors.iter().sum::<f64>() / n;
                    let max = errors.iter().cloned().fold(0.0f64, f64::max);
                    let mut sorted = errors.clone();
                    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let median = sorted[sorted.len()/2];

                    let p_x: Vec<f64> = validation_results.iter().map(|r| r.0).collect();
                    let p_y: Vec<f64> = validation_results.iter().map(|r| r.1).collect();
                    let t_x: Vec<f64> = validation_results.iter().map(|r| r.2).collect();
                    let t_y: Vec<f64> = validation_results.iter().map(|r| r.3).collect();
                    let p_xr = p_x.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - p_x.iter().cloned().fold(f64::INFINITY, f64::min);
                    let p_yr = p_y.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - p_y.iter().cloned().fold(f64::INFINITY, f64::min);
                    let t_xr = t_x.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - t_x.iter().cloned().fold(f64::INFINITY, f64::min);
                    let t_yr = t_y.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - t_y.iter().cloned().fold(f64::INFINITY, f64::min);

                    println!("\n=== CNN Multi-point validation ===");
                    println!("  Points: {}", errors.len());
                    println!("  Mean:   {mean:.0} px");
                    println!("  Median: {median:.0} px");
                    println!("  Max:    {max:.0} px");
                    println!("  Coverage X: {:.0}/{:.0} ({:.0}%)", p_xr, t_xr, 100.0 * p_xr / t_xr.max(1.0));
                    println!("  Coverage Y: {:.0}/{:.0} ({:.0}%)", p_yr, t_yr, 100.0 * p_yr / t_yr.max(1.0));
                }
                calib.finish_validation();
            }
        }

        // Running phase
        if calib.phase() == Phase::Running {
            if mouse_edge {
                if let Some((yaw, pitch)) = current_gaze {
                    mapper.add(yaw, pitch, mouse_pos.0 as f32, mouse_pos.1 as f32);
                    mapper.fit();
                    println!("  +1 sample, total {}", mapper.samples.len());
                }
            }
            if let Some((yaw, pitch)) = current_gaze {
                if let Some((px, py)) = mapper.predict(yaw, pitch) {
                    let cx = (px as f64).clamp(0.0, sw as f64 - 1.0);
                    let cy = (py as f64).clamp(0.0, sh as f64 - 1.0);
                    let t_sec = now_ms as f64 / 1000.0;
                    let (gx_f, gy_f) = one_euro.filter((cx, cy), t_sec);
                    let gx = gx_f as usize;
                    let gy = gy_f as usize;
                    draw_filled_circle(&mut buf, sw, sh, gx, gy, 10, 0x00FF00);
                    draw_ring(&mut buf, sw, sh, gx, gy, 20, 0xFFFFFF);
                    draw_ring(&mut buf, sw, sh, gx, gy, 30, 0x00FF00);
                }
            }
        }

        fps_c.tick();
        let f = fps_c.fps();
        let bar = (f as usize * 5).min(sw);
        let bc = if f > 15.0 { 0x00FF00 } else if f > 8.0 { 0xFFFF00 } else { 0xFF0000 };
        for x in 0..bar { for y in 0..4 { buf[y*sw+x] = bc; } }
        window.update_with_buffer(&buf, sw, sh).unwrap();

        if fps_c.count % 30 == 0 {
            let mode = match calib.phase() {
                Phase::Idle => "IDLE", Phase::Calibrating => "CALIB",
                Phase::Validating => "VALID", Phase::Running => "RUN",
            };
            print!("\r[{mode}] FPS:{f:.0} | samples:{} | gaze:{}    ",
                mapper.samples.len(),
                if current_gaze.is_some() { "ok" } else { "?" });
        }
    }
    println!("\nDone.");
}

fn run_gaze(model: &Model, rgb: &[u8], w: usize, h: usize, fx: usize, fy: usize, fw: usize, fh: usize) -> Option<(f32, f32)> {
    if fx+fw > w || fy+fh > h || fw < 20 || fh < 20 { return None; }
    let sz = 448;
    let mean = [0.485f32, 0.456, 0.406];
    let std = [0.229f32, 0.224, 0.225];
    let mut data = vec![0.0f32; 3*sz*sz];
    for y in 0..sz { for x in 0..sz {
        let sx = fx + x*fw/sz;
        let sy = fy + y*fh/sz;
        if sx < w && sy < h {
            let si = (sy*w+sx)*3;
            for c in 0..3 { data[c*sz*sz+y*sz+x] = (rgb[si+c] as f32 / 255.0 - mean[c]) / std[c]; }
        }
    }}
    let t = Tensor::from(tract_ndarray::Array4::from_shape_vec((1,3,sz,sz), data).ok()?).into();
    let r = model.run(tvec![t]).ok()?;
    let yaw_view = r[0].to_array_view::<f32>().ok()?;
    let pitch_view = r[1].to_array_view::<f32>().ok()?;
    let yaw_bins = yaw_view.as_slice()?;
    let pitch_bins = pitch_view.as_slice()?;
    Some((bins_to_angle(yaw_bins), bins_to_angle(pitch_bins)))
}

fn bins_to_angle(bins: &[f32]) -> f32 {
    let max_val = bins.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = bins.iter().map(|&b| (b - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().enumerate().map(|(i, &e)| (e / sum) * (i as f32 * 4.0 - 180.0)).sum()
}

// --- UI helpers ---
struct SmRect{x:f64,y:f64,w:f64,h:f64,a:f64,init:bool}
impl SmRect{fn new(a:f64)->Self{Self{x:0.0,y:0.0,w:0.0,h:0.0,a,init:false}}fn update(&mut self,x:f64,y:f64,w:f64,h:f64){if!self.init{self.x=x;self.y=y;self.w=w;self.h=h;self.init=true;}else{let a=self.a;self.x=a*x+(1.0-a)*self.x;self.y=a*y+(1.0-a)*self.y;self.w=a*w+(1.0-a)*self.w;self.h=a*h+(1.0-a)*self.h;}}fn get(&self)->(usize,usize,usize,usize){(self.x.round()as usize,self.y.round()as usize,self.w.round().max(1.0)as usize,self.h.round().max(1.0)as usize)}}
fn draw_filled_circle(b:&mut[u32],w:usize,h:usize,cx:usize,cy:usize,r:usize,c:u32){for dy in 0..=r{for dx in 0..=r{if dx*dx+dy*dy<=r*r{for&(sx,sy)in&[(cx+dx,cy+dy),(cx.wrapping_sub(dx),cy+dy),(cx+dx,cy.wrapping_sub(dy)),(cx.wrapping_sub(dx),cy.wrapping_sub(dy))]{if sx<w&&sy<h{b[sy*w+sx]=c;}}}}}}
fn draw_ring(b:&mut[u32],w:usize,h:usize,cx:usize,cy:usize,r:usize,c:u32){for i in 0..64{let t=2.0*std::f64::consts::PI*i as f64/64.0;let x=(cx as f64+r as f64*t.cos()).round()as i32;let y=(cy as f64+r as f64*t.sin()).round()as i32;if x>=0&&(x as usize)<w&&y>=0&&(y as usize)<h{b[y as usize*w+x as usize]=c;}}}
fn draw_rect(b:&mut[u32],w:usize,h:usize,x:usize,y:usize,rw:usize,rh:usize,c:u32){for dx in 0..rw{let px=x+dx;if px<w{if y<h{b[y*w+px]=c;}let by=y+rh.saturating_sub(1);if by<h{b[by*w+px]=c;}}}for dy in 0..rh{let py=y+dy;if py<h{if x<w{b[py*w+x]=c;}let bx=x+rw.saturating_sub(1);if bx<w{b[py*w+bx]=c;}}}}
struct FpsC{t:Instant,count:u64,fps:f64}
impl FpsC{fn new()->Self{Self{t:Instant::now(),count:0,fps:0.0}}fn tick(&mut self){self.count+=1;let e=self.t.elapsed().as_secs_f64();if e>=1.0{self.fps=self.count as f64/e;self.count=0;self.t=Instant::now();}}fn fps(&self)->f64{self.fps}}
