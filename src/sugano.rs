//! Sugano 2014 / ETH-XGaze 2020 head-pose normalization for gaze estimation.
//!
//! Verbatim port of `xucong-zhang/ETH-XGaze/normalization_example.py`:
//!
//! ```python
//! hRx = hR[:, 0]                              # head rotation first column
//! forward = (et / distance).reshape(3)        # unit vector cam → eye
//! down = np.cross(forward, hRx)
//! down /= np.linalg.norm(down)
//! right = np.cross(down, forward)
//! right /= np.linalg.norm(right)
//! R = np.c_[right, down, forward].T           # 3×3 rotation, camera → virtual
//!
//! S = diag(1, 1, z_scale)                     # z_scale = distance_norm / distance
//! W = cam_norm @ S @ R @ inv(cam)             # image warp matrix (3×3 homography)
//!
//! img_warped = cv2.warpPerspective(img, W, roiSize)
//!
//! # Gaze vector normalization (for training labels):
//! gc_normalized = R @ (gc - et)
//! gc_normalized /= np.linalg.norm(gc_normalized)
//! ```
//!
//! ETH-XGaze canonical parameters:
//! - Face normalization: `focal_norm=960, distance_norm=300mm, roiSize=(448, 448)`
//! - Eye  normalization: `focal_norm=1800, distance_norm=600mm, roiSize=(128, 128)`

use nalgebra::{Matrix3, Vector3};

/// Canonical ETH-XGaze face normalization parameters.
pub struct FaceNormParams {
    pub focal_norm: f64,
    pub distance_norm: f64,
    pub roi_w: usize,
    pub roi_h: usize,
}

impl FaceNormParams {
    pub const ETH_XGAZE: Self = Self {
        focal_norm: 960.0,
        distance_norm: 300.0,
        roi_w: 448,
        roi_h: 448,
    };

    pub const MPII_EYE: Self = Self {
        focal_norm: 1800.0,
        distance_norm: 600.0,
        roi_w: 128,
        roi_h: 128,
    };
}

/// Generic 3D face model — 6 anchor points in millimeters, with real depth
/// so PnP can recover pitch/yaw (not just roll). z > 0 means behind the eye plane.
/// Based on average adult face anthropometry.
pub fn face_model_3d() -> [Vector3<f64>; 6] {
    let cy = 16.666666666666668;
    [
        // Eyes protrude forward (z=0), mouth is recessed (z=+25)
        Vector3::new(-45.0, 40.0 - cy, 0.0),  // right eye outer (iBUG 36)
        Vector3::new(-30.0, 40.0 - cy, 0.0),  // right eye inner (iBUG 39)
        Vector3::new( 30.0, 40.0 - cy, 0.0),  // left eye inner  (iBUG 42)
        Vector3::new( 45.0, 40.0 - cy, 0.0),  // left eye outer  (iBUG 45)
        Vector3::new(-25.0, -30.0 - cy, 25.0), // mouth right    (iBUG 48)
        Vector3::new( 25.0, -30.0 - cy, 25.0), // mouth left     (iBUG 54)
    ]
}

/// PnP via direct linear scale + Procrustes rotation recovery.
/// Returns (rotation, translation) — rotation such that world = R·model + t.
pub fn solve_pnp(
    image_points: &[(f64, f64); 6],
    focal_length: f64,
    image_w: f64,
    image_h: f64,
) -> Option<(Matrix3<f64>, Vector3<f64>)> {
    let model = face_model_3d();
    let cx = image_w / 2.0;
    let cy = image_h / 2.0;

    // Estimate z from inter-eye distance (90 mm model)
    let img_iw = ((image_points[3].0 - image_points[0].0).powi(2)
        + (image_points[3].1 - image_points[0].1).powi(2))
    .sqrt();
    if img_iw < 1.0 {
        return None;
    }
    let z = focal_length * 90.0 / img_iw;

    // Face center in image → 3D translation
    let face_cx = image_points.iter().map(|p| p.0).sum::<f64>() / 6.0;
    let face_cy = image_points.iter().map(|p| p.1).sum::<f64>() / 6.0;
    let tx = (face_cx - cx) * z / focal_length;
    let ty = (face_cy - cy) * z / focal_length;
    let translation = Vector3::new(tx, ty, z);

    // Back-project 2D points onto z=const plane, centered at face
    let mut obs: [Vector3<f64>; 6] = [Vector3::zeros(); 6];
    let mut centroid_obs = Vector3::zeros();
    let mut centroid_model = Vector3::zeros();
    for i in 0..6 {
        let ox = (image_points[i].0 - cx) * z / focal_length - tx;
        let oy = (image_points[i].1 - cy) * z / focal_length - ty;
        obs[i] = Vector3::new(ox, oy, 0.0);
        centroid_obs += obs[i];
        centroid_model += model[i];
    }
    centroid_obs /= 6.0;
    centroid_model /= 6.0;

    // Procrustes: find R minimizing ||R·model - obs||
    let mut h = Matrix3::zeros();
    for i in 0..6 {
        let m = model[i] - centroid_model;
        let o = obs[i] - centroid_obs;
        h += m * o.transpose();
    }
    let svd = h.svd(true, true);
    let u = svd.u?;
    let v_t = svd.v_t?;
    let mut rotation = v_t.transpose() * u.transpose();
    if rotation.determinant() < 0.0 {
        let mut v = v_t.transpose();
        v.column_mut(2).neg_mut();
        rotation = v * u.transpose();
    }

    Some((rotation, translation))
}

/// Compute the ETH-XGaze normalization rotation R given head rotation and eye target vector.
/// Verbatim port of the Python snippet.
pub fn compute_normalization_rotation(
    head_rotation: &Matrix3<f64>,
    et: &Vector3<f64>, // eye target position in camera frame
) -> (Matrix3<f64>, f64) {
    let distance = et.norm();
    let forward = et / distance;
    let h_rx = head_rotation.column(0).into_owned();
    let mut down = forward.cross(&h_rx);
    let dn = down.norm();
    if dn > 1e-10 {
        down /= dn;
    }
    let mut right = down.cross(&forward);
    let rn = right.norm();
    if rn > 1e-10 {
        right /= rn;
    }
    // R = columns(right, down, forward).T
    let r = Matrix3::from_columns(&[right, down, forward]).transpose();
    (r, distance)
}

/// Build the 3×3 warp (homography) matrix W = cam_norm · S · R · inv(cam).
pub fn build_warp_matrix(
    cam: &Matrix3<f64>, // real camera intrinsic (3×3)
    cam_norm: &Matrix3<f64>, // normalized camera intrinsic
    r: &Matrix3<f64>,  // normalization rotation
    z_scale: f64, // distance_norm / distance
) -> Matrix3<f64> {
    let s = Matrix3::new(
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, z_scale,
    );
    let cam_inv = cam.try_inverse().unwrap_or_else(Matrix3::identity);
    cam_norm * s * r * cam_inv
}

/// Build a standard pinhole intrinsic matrix.
pub fn make_intrinsic(fx: f64, fy: f64, cx: f64, cy: f64) -> Matrix3<f64> {
    Matrix3::new(
        fx, 0.0, cx,
        0.0, fy, cy,
        0.0, 0.0, 1.0,
    )
}

/// Apply perspective warp using inverse mapping + bilinear sampling.
/// For each destination pixel (u, v), compute source coordinates via W⁻¹.
pub fn warp_perspective_rgb(
    src: &[u8],
    src_w: usize,
    src_h: usize,
    w_matrix: &Matrix3<f64>,
    dst_w: usize,
    dst_h: usize,
) -> Vec<u8> {
    let mut dst = vec![0u8; dst_w * dst_h * 3];
    let w_inv = match w_matrix.try_inverse() {
        Some(inv) => inv,
        None => return dst,
    };

    for dy in 0..dst_h {
        for dx in 0..dst_w {
            // Inverse warp: (dx, dy, 1) → source (sx, sy, s)
            let u = dx as f64;
            let v = dy as f64;
            let sx_h = w_inv[(0, 0)] * u + w_inv[(0, 1)] * v + w_inv[(0, 2)];
            let sy_h = w_inv[(1, 0)] * u + w_inv[(1, 1)] * v + w_inv[(1, 2)];
            let s_h  = w_inv[(2, 0)] * u + w_inv[(2, 1)] * v + w_inv[(2, 2)];
            if s_h.abs() < 1e-10 { continue; }
            let sx = sx_h / s_h;
            let sy = sy_h / s_h;

            if sx < 0.0 || sy < 0.0 || sx >= src_w as f64 - 1.0 || sy >= src_h as f64 - 1.0 {
                continue;
            }
            let x0 = sx as usize;
            let y0 = sy as usize;
            let fx = (sx - x0 as f64) as f32;
            let fy = (sy - y0 as f64) as f32;
            for c in 0..3 {
                let p00 = src[(y0 * src_w + x0) * 3 + c] as f32;
                let p01 = src[(y0 * src_w + x0 + 1) * 3 + c] as f32;
                let p10 = src[((y0 + 1) * src_w + x0) * 3 + c] as f32;
                let p11 = src[((y0 + 1) * src_w + x0 + 1) * 3 + c] as f32;
                let val = p00 * (1.0 - fx) * (1.0 - fy)
                    + p01 * fx * (1.0 - fy)
                    + p10 * (1.0 - fx) * fy
                    + p11 * fx * fy;
                dst[(dy * dst_w + dx) * 3 + c] = val as u8;
            }
        }
    }
    dst
}

/// End-to-end face normalization using ETH-XGaze recipe.
///
/// - `src_rgb`: full camera image RGB
/// - `cam`: real camera intrinsic matrix
/// - `head_rotation`: head rotation from PnP
/// - `face_center_3d`: face center in camera frame (mm)
/// - `params`: normalization parameters (focal_norm, distance_norm, roi size)
///
/// Returns (normalized_image, normalization_rotation_R) for gaze denormalization.
pub fn normalize_face(
    src_rgb: &[u8],
    src_w: usize,
    src_h: usize,
    cam: &Matrix3<f64>,
    head_rotation: &Matrix3<f64>,
    face_center_3d: &Vector3<f64>,
    params: &FaceNormParams,
) -> (Vec<u8>, Matrix3<f64>) {
    let (r, distance) = compute_normalization_rotation(head_rotation, face_center_3d);
    let z_scale = params.distance_norm / distance;

    let cam_norm = make_intrinsic(
        params.focal_norm,
        params.focal_norm,
        params.roi_w as f64 / 2.0,
        params.roi_h as f64 / 2.0,
    );

    let w = build_warp_matrix(cam, &cam_norm, &r, z_scale);
    let warped = warp_perspective_rgb(src_rgb, src_w, src_h, &w, params.roi_w, params.roi_h);
    (warped, r)
}

/// Un-rotate a predicted gaze direction from normalized camera frame back to real camera frame.
/// `gaze_norm`: (yaw_rad, pitch_rad) in normalized frame
/// `r_norm`: the normalization rotation R used during normalize_face
pub fn denormalize_gaze(yaw_norm: f32, pitch_norm: f32, r_norm: &Matrix3<f64>) -> (f32, f32) {
    // Convert (yaw, pitch) to 3D unit vector in normalized frame
    let yn = yaw_norm as f64;
    let pn = pitch_norm as f64;
    let g = Vector3::new(
        pn.cos() * yn.sin(),
        pn.sin(),
        -pn.cos() * yn.cos(),
    );
    // Un-rotate: real = R^T · normalized
    let g_real = r_norm.transpose() * g;
    let yaw_real = g_real.x.atan2(-g_real.z);
    let pitch_real = g_real.y.asin();
    (yaw_real as f32, pitch_real as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pnp_recovers_identity_rotation() {
        let model = face_model_3d();
        let focal = 1000.0;
        let img_w = 1280.0;
        let img_h = 720.0;
        let cx = img_w / 2.0;
        let cy = img_h / 2.0;
        let tz = 500.0;

        let mut img_pts = [(0.0f64, 0.0f64); 6];
        for i in 0..6 {
            let p = model[i] + Vector3::new(0.0, 0.0, tz);
            img_pts[i] = (cx + p.x * focal / p.z, cy + p.y * focal / p.z);
        }

        let (r, t) = solve_pnp(&img_pts, focal, img_w, img_h).expect("pnp ok");
        assert!((t.z - tz).abs() < 50.0, "tz={}", t.z);
        let trace = r.trace();
        assert!((trace - 3.0).abs() < 0.5, "trace={trace}");
    }

    #[test]
    fn normalization_rotation_aligns_forward() {
        // When head is identity and eye is directly in front (0,0,500)
        let hr = Matrix3::identity();
        let et = Vector3::new(0.0, 0.0, 500.0);
        let (r, dist) = compute_normalization_rotation(&hr, &et);
        assert!((dist - 500.0).abs() < 1e-6);
        // forward = (0,0,1); after R, (0,0,500) should become (0,0,500)
        let transformed = r * et;
        assert!(transformed.x.abs() < 1e-6);
        assert!(transformed.y.abs() < 1e-6);
        assert!((transformed.z - 500.0).abs() < 1e-6);
    }

    #[test]
    fn warp_identity_matrix_preserves_image() {
        let src = vec![100u8; 10 * 10 * 3];
        let w = Matrix3::identity();
        let dst = warp_perspective_rgb(&src, 10, 10, &w, 10, 10);
        // Identity warp should preserve most of the image
        let non_zero = dst.iter().filter(|&&v| v > 0).count();
        assert!(non_zero > 200, "warp lost data: {non_zero}");
    }

    #[test]
    fn build_warp_matrix_is_3x3() {
        let cam = make_intrinsic(1000.0, 1000.0, 640.0, 360.0);
        let cam_norm = make_intrinsic(960.0, 960.0, 224.0, 224.0);
        let r = Matrix3::identity();
        let w = build_warp_matrix(&cam, &cam_norm, &r, 0.6);
        // Sanity check: result shouldn't be all zeros
        assert!(w.norm() > 0.1);
    }

    #[test]
    fn face_norm_params_constants() {
        assert_eq!(FaceNormParams::ETH_XGAZE.focal_norm, 960.0);
        assert_eq!(FaceNormParams::ETH_XGAZE.distance_norm, 300.0);
        assert_eq!(FaceNormParams::MPII_EYE.focal_norm, 1800.0);
        assert_eq!(FaceNormParams::MPII_EYE.distance_norm, 600.0);
    }
}
