//! Real-time eye tracking from webcam (tract backend — WASM-compatible inference).
//!
//! Pipeline: nokhwa (camera) → rustface (face bbox) → PFLD/tract (68 landmarks)
//!          → OCEC/tract (eye open/closed) → Timm & Barth (pupil) → MobileGaze/tract (gaze)
//!
//! Usage:
//!   cargo run --release --features demo --example webcam

use minifb::{Key, Window, WindowOptions};
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{CameraIndex, RequestedFormat, RequestedFormatType};
use nokhwa::Camera;
use rustface::ImageData;
use saccade::blink::{BlinkDetector, EyeState};
use saccade::classify::{EyeEvent, IVTClassifier};
use saccade::frame::GrayFrame;
use saccade::timm::{self, TimmConfig};
use std::time::Instant;
use tract_onnx::prelude::*;

type TractModel = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

const FACE_MODEL: &str = "seeta_fd_frontal_v1.0.bin";
const FACE_URL: &str = "https://github.com/atomashpolskiy/rustface/raw/master/model/seeta_fd_frontal_v1.0.bin";
const PFLD_MODEL: &str = "pfld.onnx";
const PFLD_URL: &str = "https://github.com/cunjian/pytorch_face_landmark/raw/refs/heads/master/onnx/pfld.onnx";
const OCEC_MODEL: &str = "ocec_m.onnx";
const OCEC_URL: &str = "https://github.com/PINTO0309/OCEC/releases/download/onnx/ocec_m.onnx";
const GAZE_MODEL: &str = "mobileone_s0_gaze.onnx";
const GAZE_URL: &str = "https://github.com/yakhyo/gaze-estimation/releases/download/weights/mobileone_s0_gaze.onnx";

fn download(path: &str, url: &str) {
    if !std::path::Path::new(path).exists() {
        println!("Downloading {path}...");
        let ok = std::process::Command::new("curl").args(["-L", "-o", path, url]).status().map(|s| s.success()).unwrap_or(false);
        if !ok { eprintln!("Failed: curl -L -o {path} {url}"); std::process::exit(1); }
    }
}

fn load_model(path: &str) -> TractModel {
    tract_onnx::onnx()
        .model_for_path(path).unwrap_or_else(|e| panic!("Failed to load {path}: {e}"))
        .into_optimized().unwrap_or_else(|e| panic!("Failed to optimize {path}: {e}"))
        .into_runnable().unwrap_or_else(|e| panic!("Failed to make {path} runnable: {e}"))
}

fn main() {
    download(FACE_MODEL, FACE_URL);
    download(PFLD_MODEL, PFLD_URL);
    download(OCEC_MODEL, OCEC_URL);
    download(GAZE_MODEL, GAZE_URL);

    let mut face_det = rustface::create_detector(FACE_MODEL).expect("face model");
    face_det.set_min_face_size(80);
    face_det.set_score_thresh(2.0);

    println!("Loading ONNX models via tract...");
    let pfld = load_model(PFLD_MODEL);
    let ocec = load_model(OCEC_MODEL);
    let gaze_net = load_model(GAZE_MODEL);
    println!("Models loaded.");

    let format = RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);
    let mut camera = Camera::new(CameraIndex::Index(0), format).expect("camera");
    camera.open_stream().expect("stream");

    let cam_res = camera.resolution();
    let scale_down = (cam_res.width() / 640).max(1) as usize;
    let cam_w = cam_res.width() as usize / scale_down;
    let cam_h = cam_res.height() as usize / scale_down;
    println!("Camera: {}×{} → {}×{}", cam_res.width(), cam_res.height(), cam_w, cam_h);

    let mut window = Window::new("Saccade — Eye Tracker (ESC to quit)", cam_w, cam_h, WindowOptions::default()).unwrap();
    window.set_target_fps(60);

    let timm_cfg = TimmConfig { gradient_threshold: 0.2, use_weight_map: true, weight_blur_sigma: 2.0 };
    let mut face_sm = SmRect::new(0.25);
    let mut lp_sm = SmPt::new(0.4);
    let mut rp_sm = SmPt::new(0.4);
    let mut l_open_sm = 1.0f64;
    let mut r_open_sm = 1.0f64;
    let mut l_blink = BlinkDetector::new(); l_blink.confidence_threshold = 0.15;
    let mut r_blink = BlinkDetector::new(); r_blink.confidence_threshold = 0.15;
    let mut l_ocec_roi = OcecEyeState::new();
    let mut r_ocec_roi = OcecEyeState::new();
    let mut l_baseline_sum = 0.0f64;
    let mut r_baseline_sum = 0.0f64;
    let mut cal_frames = 0u32;
    let mut classifier = IVTClassifier::default_params();
    let start = Instant::now();
    let mut buf = vec![0u32; cam_w * cam_h];
    let mut fps = FpsC::new();
    let mut no_face = 0u32;
    let mut gaze_yaw = 0.0f64;
    let mut gaze_pitch = 0.0f64;

    println!("Running... ESC to quit.");

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let now_ms = start.elapsed().as_millis() as u64;

        let decoded = match camera.frame() {
            Ok(f) => match f.decode_image::<RgbFormat>() { Ok(i) => i, Err(_) => continue },
            Err(_) => continue,
        };
        let (fw, fh) = (decoded.width() as usize, decoded.height() as usize);
        let rgb_full = decoded.as_raw();

        let mut rgb = vec![0u8; cam_w * cam_h * 3];
        let mut gray = vec![0u8; cam_w * cam_h];
        for y in 0..cam_h { for x in 0..cam_w {
            let (sx, sy) = (x * scale_down, y * scale_down);
            if sx < fw && sy < fh {
                let si = (sy * fw + sx) * 3;
                let (r, g, b) = (rgb_full[si], rgb_full[si+1], rgb_full[si+2]);
                let di = (y * cam_w + x) * 3;
                rgb[di] = r; rgb[di+1] = g; rgb[di+2] = b;
                gray[y * cam_w + x] = (0.299 * r as f32 + 0.587 * g as f32 + 0.114 * b as f32) as u8;
            }
        }}

        let faces = face_det.detect(&ImageData::new(&gray, cam_w as u32, cam_h as u32));
        for i in 0..cam_w*cam_h { buf[i] = ((rgb[i*3] as u32)<<16)|((rgb[i*3+1] as u32)<<8)|rgb[i*3+2] as u32; }

        let best = faces.iter().max_by_key(|f| { let b=f.bbox(); b.width()*b.height() });

        if let Some(face) = best {
            no_face = 0;
            let bb = face.bbox();
            face_sm.update(bb.x().max(0) as f64, bb.y().max(0) as f64, bb.width() as f64, bb.height() as f64);
            let (fx, fy, fw, fh) = face_sm.get();
            draw_rect(&mut buf, cam_w, cam_h, fx, fy, fw, fh, 0xFFFF00);

            if let Some(lm) = run_pfld(&pfld, &gray, cam_w, cam_h, fx, fy, fw, fh) {
                for &(lx, ly) in &lm { let (px, py) = (lx as usize, ly as usize);
                    if px < cam_w && py < cam_h { buf[py*cam_w+px] = 0x00FF00; }
                }

                // OCEC
                let r_open = run_ocec(&ocec, &gray, cam_w, cam_h, &lm, 36, 41, &mut r_ocec_roi);
                let l_open = run_ocec(&ocec, &gray, cam_w, cam_h, &lm, 42, 47, &mut l_ocec_roi);

                l_open_sm = 0.5 * l_open as f64 + 0.5 * l_open_sm;
                r_open_sm = 0.5 * r_open as f64 + 0.5 * r_open_sm;

                if cal_frames < 45 {
                    l_baseline_sum += l_open_sm;
                    r_baseline_sum += r_open_sm;
                    cal_frames += 1;
                    if cal_frames == 45 {
                        let l_base = l_baseline_sum / 45.0;
                        let r_base = r_baseline_sum / 45.0;
                        l_blink.confidence_threshold = (l_base * 0.4).max(0.03);
                        r_blink.confidence_threshold = (r_base * 0.4).max(0.03);
                        eprintln!("\nCalibrated: L_base={l_base:.3} thresh={:.3} | R_base={r_base:.3} thresh={:.3}",
                            l_blink.confidence_threshold, r_blink.confidence_threshold);
                    }
                }

                let l_state = l_blink.update(l_open_sm, now_ms);
                let r_state = r_blink.update(r_open_sm, now_ms);

                draw_filled_circle(&mut buf, cam_w, cam_h, cam_w-40, 15, 8, state_color(l_state));
                draw_filled_circle(&mut buf, cam_w, cam_h, cam_w-20, 15, 8, state_color(r_state));

                // Pupils
                if l_open_sm > 0.5 {
                    if let Some((px, py)) = detect_pupil(&lm, 42, 47, &gray, cam_w, cam_h, &timm_cfg) {
                        lp_sm.update(px, py);
                        let (sx, sy) = lp_sm.get();
                        draw_cross(&mut buf, cam_w, cam_h, sx, sy, 0x00FFFF);
                    }
                }
                if r_open_sm > 0.5 {
                    if let Some((px, py)) = detect_pupil(&lm, 36, 41, &gray, cam_w, cam_h, &timm_cfg) {
                        rp_sm.update(px, py);
                        let (sx, sy) = rp_sm.get();
                        draw_cross(&mut buf, cam_w, cam_h, sx, sy, 0xFF0000);
                        classifier.update(px, py, now_ms);
                    }
                }

                // Gaze
                if let Some((yaw, pitch)) = run_gaze(&gaze_net, &rgb, cam_w, cam_h, fx, fy, fw, fh) {
                    gaze_yaw = 0.3 * yaw as f64 + 0.7 * gaze_yaw;
                    gaze_pitch = 0.3 * pitch as f64 + 0.7 * gaze_pitch;
                    let cx = fx + fw / 2;
                    let cy = fy + fh / 2;
                    let al = 60.0;
                    let ax = cx as f64 - al * gaze_yaw.to_radians().sin();
                    let ay = cy as f64 - al * gaze_pitch.to_radians().sin();
                    draw_line(&mut buf, cam_w, cam_h, cx, cy, ax as usize, ay as usize, 0xFF00FF);
                }
            }
        } else {
            no_face += 1;
            if no_face > 30 { l_blink.reset(); r_blink.reset(); lp_sm = SmPt::new(0.4); rp_sm = SmPt::new(0.4); no_face = 0; }
            else if face_sm.init { let (fx,fy,fw,fh)=face_sm.get(); draw_rect(&mut buf,cam_w,cam_h,fx,fy,fw,fh,0x666600); }
        }

        fps.tick();
        let f = fps.fps();
        let bar = (f as usize * 3).min(cam_w);
        let bc = if f > 15.0 { 0x00FF00 } else if f > 8.0 { 0xFFFF00 } else { 0xFF0000 };
        for x in 0..bar { for y in 0..4 { buf[y*cam_w+x] = bc; } }
        window.update_with_buffer(&buf, cam_w, cam_h).unwrap();

        if fps.count % 30 == 0 {
            let ls = format!("{:?}", l_blink.state());
            let rs = format!("{:?}", r_blink.state());
            let bl = l_blink.blink_count() + r_blink.blink_count();
            let bpm = (l_blink.blinks_per_minute(now_ms, 60000) + r_blink.blinks_per_minute(now_ms, 60000)) / 2.0;
            let fix = classifier.events().iter().filter(|e| matches!(e, EyeEvent::Fixation(_))).count();
            let sac = classifier.events().iter().filter(|e| matches!(e, EyeEvent::Saccade(_))).count();
            print!("\rFPS:{f:.0} | Eyes:{ls}/{rs} | Blinks:{bl}({bpm:.0}/m) | Gaze:y={gaze_yaw:.1}° p={gaze_pitch:.1}° | Fix:{fix} Sac:{sac}    ");
        }
    }
    println!("\nDone.");
}

// --- Tract model runners ---

fn run_pfld(model: &TractModel, g: &[u8], w: usize, h: usize, fx: usize, fy: usize, fw: usize, fh: usize) -> Option<Vec<(f32,f32)>> {
    if fx+fw>w || fy+fh>h || fw<20 || fh<20 { return None; }
    let sz = 112;
    let mut data = vec![0.0f32; 3*sz*sz];
    for y in 0..sz { for x in 0..sz {
        let (sx, sy) = (fx + x*fw/sz, fy + y*fh/sz);
        let v = if sx<w && sy<h { g[sy*w+sx] as f32 / 255.0 } else { 0.0 };
        data[y*sz+x] = v; data[sz*sz+y*sz+x] = v; data[2*sz*sz+y*sz+x] = v;
    }}
    let tensor = Tensor::from(tract_ndarray::Array4::from_shape_vec((1, 3, sz, sz), data).ok()?).into();
    let result = model.run(tvec![tensor]).ok()?;
    let out = result[0].to_array_view::<f32>().ok()?;
    let flat = out.as_slice()?;
    if flat.len() < 136 { return None; }
    Some((0..68).map(|i| (flat[i*2]*fw as f32 + fx as f32, flat[i*2+1]*fh as f32 + fy as f32)).collect())
}

fn run_ocec(model: &TractModel, g: &[u8], w: usize, h: usize, lm: &[(f32,f32)], si: usize, ei: usize, roi_state: &mut OcecEyeState) -> f32 {
    let pts = &lm[si..=ei];
    let min_x = pts.iter().map(|p| p.0).fold(f32::MAX, f32::min);
    let max_x = pts.iter().map(|p| p.0).fold(f32::MIN, f32::max);
    let min_y = pts.iter().map(|p| p.1).fold(f32::MAX, f32::min);
    let max_y = pts.iter().map(|p| p.1).fold(f32::MIN, f32::max);
    let mx = (max_x - min_x) * 0.15;
    let my = (max_y - min_y) * 0.3;
    let new_rx = (min_x - mx).max(0.0) as f64;
    let new_ry = (min_y - my).max(0.0) as f64;
    let new_rw = ((max_x - min_x) + 2.0*mx).max(4.0) as f64;
    let new_rh = ((max_y - min_y) + 2.0*my).max(4.0) as f64;
    if !roi_state.frozen { roi_state.update(new_rx, new_ry, new_rw, new_rh); }
    let (rx, ry, rw, rh) = roi_state.get();
    if rx+rw>w || ry+rh>h { return 1.0; }

    let (oh, ow) = (24, 40);
    let mut data = vec![0.0f32; 3*oh*ow];
    for y in 0..oh { for x in 0..ow {
        let sx = rx + x*rw/ow;
        let sy = ry + y*rh/oh;
        let v = if sx<w && sy<h { g[sy*w+sx] as f32 / 255.0 } else { 0.0 };
        data[y*ow+x] = v; data[oh*ow+y*ow+x] = v; data[2*oh*ow+y*ow+x] = v;
    }}
    let tensor = Tensor::from(tract_ndarray::Array4::from_shape_vec((1, 3, oh, ow), data).unwrap()).into();
    let result = match model.run(tvec![tensor]) { Ok(r) => r, Err(_) => return 1.0 };
    let val = result[0].to_array_view::<f32>().ok().and_then(|a| a.as_slice().map(|s| if s.is_empty() { 1.0 } else { s[0] })).unwrap_or(1.0);
    if val < 0.3 { roi_state.freeze(); } else { roi_state.unfreeze(); }
    val
}

fn run_gaze(model: &TractModel, rgb: &[u8], w: usize, h: usize, fx: usize, fy: usize, fw: usize, fh: usize) -> Option<(f32,f32)> {
    if fx+fw>w || fy+fh>h || fw<20 || fh<20 { return None; }
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
    let tensor = Tensor::from(tract_ndarray::Array4::from_shape_vec((1, 3, sz, sz), data).ok()?).into();
    let result = model.run(tvec![tensor]).ok()?;

    let yaw_view = result[0].to_array_view::<f32>().ok()?;
    let pitch_view = result[1].to_array_view::<f32>().ok()?;
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

fn detect_pupil(lm: &[(f32,f32)], si: usize, ei: usize, g: &[u8], w: usize, h: usize, cfg: &TimmConfig) -> Option<(f64,f64)> {
    let pts = &lm[si..=ei];
    let (mnx,mxx) = (pts.iter().map(|p|p.0).fold(f32::MAX,f32::min), pts.iter().map(|p|p.0).fold(f32::MIN,f32::max));
    let (mny,mxy) = (pts.iter().map(|p|p.1).fold(f32::MAX,f32::min), pts.iter().map(|p|p.1).fold(f32::MIN,f32::max));
    let (mx, my) = ((mxx-mnx)*0.3, (mxy-mny)*0.5);
    let (rx,ry) = ((mnx-mx).max(0.0) as usize, (mny-my).max(0.0) as usize);
    let (rw,rh) = (((mxx-mnx)+2.0*mx) as usize, ((mxy-mny)+2.0*my) as usize);
    if rx+rw>w || ry+rh>h || rw<8 || rh<6 { return None; }
    let mut d = vec![0u8; rw*rh];
    for y in 0..rh { d[y*rw..(y+1)*rw].copy_from_slice(&g[(ry+y)*w+rx..(ry+y)*w+rx+rw]); }
    let (mn,mx_v) = (*d.iter().min()?, *d.iter().max()?);
    if mx_v > mn+10 { let r=(mx_v-mn) as f32; for p in &mut d { *p=((*p as f32-mn as f32)/r*255.0) as u8; } }
    let f = GrayFrame::new(rw as u32, rh as u32, &d);
    let res = timm::detect_center(&f, cfg);
    Some((rx as f64+res.x, ry as f64+res.y))
}

// --- UI ---
struct OcecEyeState { rx:f64,ry:f64,rw:f64,rh:f64,init:bool,frozen:bool }
impl OcecEyeState {
    fn new()->Self{Self{rx:0.0,ry:0.0,rw:0.0,rh:0.0,init:false,frozen:false}}
    fn update(&mut self,rx:f64,ry:f64,rw:f64,rh:f64){if!self.init{self.rx=rx;self.ry=ry;self.rw=rw;self.rh=rh;self.init=true;}else{let a=0.3;self.rx=a*rx+(1.0-a)*self.rx;self.ry=a*ry+(1.0-a)*self.ry;self.rw=a*rw+(1.0-a)*self.rw;self.rh=a*rh+(1.0-a)*self.rh;}}
    fn get(&self)->(usize,usize,usize,usize){(self.rx.round() as usize,self.ry.round() as usize,self.rw.round().max(4.0) as usize,self.rh.round().max(4.0) as usize)}
    fn freeze(&mut self){self.frozen=true;} fn unfreeze(&mut self){self.frozen=false;}
}
struct SmRect{x:f64,y:f64,w:f64,h:f64,a:f64,init:bool}
impl SmRect{fn new(a:f64)->Self{Self{x:0.0,y:0.0,w:0.0,h:0.0,a,init:false}}fn update(&mut self,x:f64,y:f64,w:f64,h:f64){if!self.init{self.x=x;self.y=y;self.w=w;self.h=h;self.init=true;}else{let a=self.a;self.x=a*x+(1.0-a)*self.x;self.y=a*y+(1.0-a)*self.y;self.w=a*w+(1.0-a)*self.w;self.h=a*h+(1.0-a)*self.h;}}fn get(&self)->(usize,usize,usize,usize){(self.x.round() as usize,self.y.round() as usize,self.w.round().max(1.0) as usize,self.h.round().max(1.0) as usize)}}
struct SmPt{x:f64,y:f64,a:f64,init:bool}
impl SmPt{fn new(a:f64)->Self{Self{x:0.0,y:0.0,a,init:false}}fn update(&mut self,x:f64,y:f64){if!self.init{self.x=x;self.y=y;self.init=true;}else{let a=self.a;self.x=a*x+(1.0-a)*self.x;self.y=a*y+(1.0-a)*self.y;}}fn get(&self)->(usize,usize){(self.x.round() as usize,self.y.round() as usize)}}
fn state_color(s:EyeState)->u32{match s{EyeState::Open=>0x00FF00,EyeState::Blinking=>0xFFFF00,EyeState::Closed=>0xFF0000}}
fn draw_filled_circle(b:&mut[u32],w:usize,h:usize,cx:usize,cy:usize,r:usize,c:u32){for dy in 0..=r{for dx in 0..=r{if dx*dx+dy*dy<=r*r{for&(sx,sy)in&[(cx+dx,cy+dy),(cx.wrapping_sub(dx),cy+dy),(cx+dx,cy.wrapping_sub(dy)),(cx.wrapping_sub(dx),cy.wrapping_sub(dy))]{if sx<w&&sy<h{b[sy*w+sx]=c;}}}}}}
fn draw_rect(b:&mut[u32],w:usize,h:usize,x:usize,y:usize,rw:usize,rh:usize,c:u32){for dx in 0..rw{let px=x+dx;if px<w{if y<h{b[y*w+px]=c;}let by=y+rh.saturating_sub(1);if by<h{b[by*w+px]=c;}}}for dy in 0..rh{let py=y+dy;if py<h{if x<w{b[py*w+x]=c;}let bx=x+rw.saturating_sub(1);if bx<w{b[py*w+bx]=c;}}}}
fn draw_cross(b:&mut[u32],w:usize,h:usize,cx:usize,cy:usize,c:u32){for d in -6i32..=6{let x=cx as i32+d;if x>=0&&(x as usize)<w&&cy<h{b[cy*w+x as usize]=c;}let y=cy as i32+d;if y>=0&&(y as usize)<h&&cx<w{b[y as usize*w+cx]=c;}}for i in 0..48{let t=2.0*std::f64::consts::PI*i as f64/48.0;let x=(cx as f64+8.0*t.cos()).round() as i32;let y=(cy as f64+8.0*t.sin()).round() as i32;if x>=0&&(x as usize)<w&&y>=0&&(y as usize)<h{b[y as usize*w+x as usize]=c;}}}
fn draw_line(b:&mut[u32],w:usize,h:usize,x0:usize,y0:usize,x1:usize,y1:usize,c:u32){let s=((x1 as i32-x0 as i32).abs().max((y1 as i32-y0 as i32).abs())).max(1) as usize;for i in 0..=s{let x=x0 as f64+(x1 as f64-x0 as f64)*i as f64/s as f64;let y=y0 as f64+(y1 as f64-y0 as f64)*i as f64/s as f64;let(px,py)=(x.round() as usize,y.round() as usize);if px<w&&py<h{b[py*w+px]=c;if px+1<w{b[py*w+px+1]=c;}}}}
struct FpsC{t:Instant,count:u64,fps:f64}
impl FpsC{fn new()->Self{Self{t:Instant::now(),count:0,fps:0.0}}fn tick(&mut self){self.count+=1;let e=self.t.elapsed().as_secs_f64();if e>=1.0{self.fps=self.count as f64/e;self.count=0;self.t=Instant::now();}}fn fps(&self)->f64{self.fps}}
