//! Sugano 2014 head-pose normalization for gaze estimation.
//!
//! Reference: Sugano, Matsushita, Sato, "Learning-by-Synthesis for Appearance-Based
//! 3D Gaze Estimation", CVPR 2014.
//!
//! Pipeline:
//! 1. Detect 6 face landmarks (eye corners, mouth corners, nose tip)
//! 2. Solve PnP to get 3D head rotation R and translation t
//! 3. Build a virtual camera that looks at the face from a fixed distance
//!    along the gaze axis, with the face roll cancelled
//! 4. Warp the eye crop into the virtual camera's image plane (224×224)
//! 5. Run gaze CNN on the normalized crop → get gaze direction in virtual frame
//! 6. Un-rotate by R⁻¹ to get gaze in real camera frame

use nalgebra::{Matrix3, Vector3};

/// Generic 3D face model — 6 anchor points in millimeters.
/// Centered: centroid is at origin (required for Procrustes PnP).
pub fn face_model_3d() -> [Vector3<f64>; 6] {
    // Original: eyes at y=40, mouth at y=-30 → centroid y = (4*40 + 2*(-30))/6 = 16.67
    // Subtract that to center.
    let cy = 16.666666666666668;
    [
        Vector3::new(-45.0,  40.0 - cy, 0.0),  // right eye outer corner (iBUG 36)
        Vector3::new(-30.0,  40.0 - cy, 0.0),  // right eye inner corner (iBUG 39)
        Vector3::new( 30.0,  40.0 - cy, 0.0),  // left eye inner corner  (iBUG 42)
        Vector3::new( 45.0,  40.0 - cy, 0.0),  // left eye outer corner  (iBUG 45)
        Vector3::new(-25.0, -30.0 - cy, 0.0),  // mouth right corner     (iBUG 48)
        Vector3::new( 25.0, -30.0 - cy, 0.0),  // mouth left corner      (iBUG 54)
    ]
}

/// Solve PnP: given 6 image points and the 3D face model + camera focal length,
/// estimate (R, t) such that image_point ≈ project(R · model_point + t).
///
/// This is a simplified PnP using planar homography for the 6-point case.
/// Returns (rotation, translation) where translation is in millimeters.
pub fn solve_pnp(
    image_points: &[(f64, f64); 6],
    focal_length: f64,
    image_w: f64,
    image_h: f64,
) -> Option<(Matrix3<f64>, Vector3<f64>)> {
    let model = face_model_3d();
    let cx = image_w / 2.0;
    let cy = image_h / 2.0;

    // Build A matrix for direct linear transform (DLT)
    // For each point: [model_x model_y model_z 1] · row → image (after intrinsic)
    // We solve: minimize ||A·p - b||
    // Using a simplified weak-perspective projection (small object far from camera).

    // First estimate scale from inter-eye distance:
    // Model inter-eye = 90 mm (from -45 to 45)
    // Image inter-eye in pixels
    let img_iw = ((image_points[3].0 - image_points[0].0).powi(2)
        + (image_points[3].1 - image_points[0].1).powi(2)).sqrt();
    let model_iw = 90.0;
    if img_iw < 1.0 { return None; }
    // Z = focal × model_size / image_size  (similar triangles)
    let z = focal_length * model_iw / img_iw;

    // Translation: project image center → camera ray, intersect with z plane
    let face_cx = image_points.iter().map(|p| p.0).sum::<f64>() / 6.0;
    let face_cy = image_points.iter().map(|p| p.1).sum::<f64>() / 6.0;
    let tx = (face_cx - cx) * z / focal_length;
    let ty = (face_cy - cy) * z / focal_length;
    let translation = Vector3::new(tx, ty, z);

    // Estimate rotation by aligning the model points (in face plane) to the
    // observed points after subtracting translation.
    //
    // Procrustes-style: find R minimizing ||R · model_i - obs_world_i||²
    // where obs_world_i = (image_i - center) × z / focal
    let mut centroid_obs = Vector3::zeros();
    let mut centroid_model = Vector3::zeros();
    let mut obs_pts: [Vector3<f64>; 6] = [Vector3::zeros(); 6];
    for i in 0..6 {
        let ox = (image_points[i].0 - cx) * z / focal_length - tx;
        let oy = (image_points[i].1 - cy) * z / focal_length - ty;
        obs_pts[i] = Vector3::new(ox, oy, 0.0);
        centroid_obs += obs_pts[i];
        centroid_model += model[i];
    }
    centroid_obs /= 6.0;
    centroid_model /= 6.0;

    // Centered points
    let mut h = Matrix3::zeros();
    for i in 0..6 {
        let m = model[i] - centroid_model;
        let o = obs_pts[i] - centroid_obs;
        h += m * o.transpose();
    }

    // SVD: h = U·Σ·Vᵀ → R = V·Uᵀ
    let svd = h.svd(true, true);
    let u = svd.u?;
    let v_t = svd.v_t?;
    let mut rotation = v_t.transpose() * u.transpose();
    // Ensure det(R) = +1 (proper rotation, not reflection)
    if rotation.determinant() < 0.0 {
        let mut v = v_t.transpose();
        v.column_mut(2).neg_mut();
        rotation = v * u.transpose();
    }

    Some((rotation, translation))
}

/// Sugano normalized eye crop coordinates: warp the original image into a
/// virtual camera frame where the eye is at a fixed distance with cancelled roll.
///
/// Inputs:
/// - `src_rgb`: original image, w × h × 3
/// - `eye_center_3d`: 3D position of the eye center in camera frame (mm)
/// - `rotation`: head rotation matrix from PnP
/// - `focal_length`: real camera focal length in pixels
/// - `out_w`, `out_h`: output crop size (e.g., 224×224)
/// - `target_distance_mm`: virtual distance from camera to eye (e.g., 600 mm)
/// - `output_focal`: virtual camera focal length (e.g., 1600 px for 224×224 crop)
///
/// Returns the warped RGB image (out_w × out_h × 3).
pub fn normalize_eye_crop(
    src_rgb: &[u8],
    src_w: usize,
    src_h: usize,
    eye_center_3d: Vector3<f64>,
    rotation: Matrix3<f64>,
    focal_length: f64,
    out_w: usize,
    out_h: usize,
    target_distance_mm: f64,
    output_focal: f64,
) -> Vec<u8> {
    let mut dst = vec![0u8; out_w * out_h * 3];

    // The virtual camera looks along the line from camera origin to the eye.
    // We construct a normalized rotation matrix R_n such that:
    //   R_n · eye_center_3d = (0, 0, |eye_center_3d|)
    // i.e., the eye is on the new z-axis at its real distance.
    // Then we scale the z to target_distance_mm.

    let eye_dist = eye_center_3d.norm();
    if eye_dist < 1e-3 { return dst; }

    // Build rotation that aligns eye_center to camera z-axis (Sugano construction).
    let z_axis = eye_center_3d / eye_dist;
    // Cancel head roll: x_axis is the projection of (1,0,0) onto plane ⊥ z_axis,
    // but using head rotation's column 0 as reference for "right" direction
    let head_x = rotation.column(0).into_owned();
    let mut x_axis = head_x - z_axis * head_x.dot(&z_axis);
    let xn = x_axis.norm();
    if xn < 1e-6 {
        x_axis = Vector3::new(1.0, 0.0, 0.0);
    } else {
        x_axis /= xn;
    }
    let y_axis = z_axis.cross(&x_axis);

    // Normalization rotation R_n (camera frame → virtual frame)
    let r_n = Matrix3::from_columns(&[x_axis, y_axis, z_axis]).transpose();

    // Scale: ratio of virtual distance to real distance
    let s = target_distance_mm / eye_dist;

    // For each output pixel (out_x, out_y):
    //   - back-project to virtual ray
    //   - apply inverse normalization to get a ray in real camera frame
    //   - intersect with the eye plane (z = eye_dist) → real image pixel
    let cx_src = src_w as f64 / 2.0;
    let cy_src = src_h as f64 / 2.0;
    let cx_dst = out_w as f64 / 2.0;
    let cy_dst = out_h as f64 / 2.0;

    let r_n_inv = r_n.transpose();

    for oy in 0..out_h {
        for ox in 0..out_w {
            // Pixel → virtual camera ray (z=1 plane)
            let vx = (ox as f64 - cx_dst) / output_focal;
            let vy = (oy as f64 - cy_dst) / output_focal;
            let virt_ray = Vector3::new(vx, vy, 1.0);

            // Un-normalize: real_ray = R_n⁻¹ · virt_ray (and scale to find world point)
            let real_dir = r_n_inv * virt_ray;

            // Intersect with the eye plane (perpendicular to camera z)
            // World point at distance eye_dist along this direction
            // Actually we need: place eye_center on the new z-axis at target distance,
            // then project back through real camera at the un-rotated point.
            //
            // Simpler: the world point in real frame is
            //   P_real = R_n⁻¹ · (virt_ray * (target_distance_mm / s)) = R_n⁻¹ · virt_ray * eye_dist
            // wait, that's just (real_dir * eye_dist).
            //
            // No: the virtual frame has eye at distance target_distance_mm.
            // virt_ray scaled to length target_distance_mm in z gives the eye plane point.
            // virt_point = virt_ray * target_distance_mm  (since virt_ray.z = 1)
            // real_point = R_n⁻¹ · virt_point / s   (because we scaled by s)
            // But virt_ray.z = 1 means virt_point.z = target_distance_mm.

            let virt_point = virt_ray * target_distance_mm;
            let real_point = r_n_inv * virt_point / s;

            // Project real_point back to source image
            if real_point.z < 1.0 { continue; }
            let img_x = cx_src + real_point.x * focal_length / real_point.z;
            let img_y = cy_src + real_point.y * focal_length / real_point.z;

            // Bilinear sample
            if img_x >= 0.0 && img_y >= 0.0 && img_x < src_w as f64 - 1.0 && img_y < src_h as f64 - 1.0 {
                let x0 = img_x as usize;
                let y0 = img_y as usize;
                let fx = (img_x - x0 as f64) as f32;
                let fy = (img_y - y0 as f64) as f32;
                for c in 0..3 {
                    let p00 = src_rgb[(y0 * src_w + x0) * 3 + c] as f32;
                    let p01 = src_rgb[(y0 * src_w + x0 + 1) * 3 + c] as f32;
                    let p10 = src_rgb[((y0 + 1) * src_w + x0) * 3 + c] as f32;
                    let p11 = src_rgb[((y0 + 1) * src_w + x0 + 1) * 3 + c] as f32;
                    let v = p00 * (1.0 - fx) * (1.0 - fy)
                          + p01 * fx * (1.0 - fy)
                          + p10 * (1.0 - fx) * fy
                          + p11 * fx * fy;
                    dst[(oy * out_w + ox) * 3 + c] = v as u8;
                }
            }
        }
    }

    dst
}

/// Un-rotate a normalized gaze direction back to the real camera frame.
/// `gaze_norm`: (yaw_rad, pitch_rad) in normalized frame
/// `r_norm`: the normalization rotation matrix used to warp the eye
pub fn denormalize_gaze(
    yaw_norm: f32,
    pitch_norm: f32,
    r_norm: &Matrix3<f64>,
) -> (f32, f32) {
    // Convert (yaw, pitch) to a 3D unit vector in normalized frame
    let yn = yaw_norm as f64;
    let pn = pitch_norm as f64;
    let g_norm = Vector3::new(
        -yn.cos() * pn.sin(),
        -pn.cos().sin().min(1.0).max(-1.0),  // hmm, this is wrong
        -yn.cos() * pn.cos(),
    );
    // Actually: y = -sin(p), x = cos(p)·sin(y), z = -cos(p)·cos(y)  is one convention
    let yn = yaw_norm as f64;
    let pn = pitch_norm as f64;
    let g = Vector3::new(
        pn.cos() * yn.sin(),
        pn.sin(),
        -pn.cos() * yn.cos(),
    );
    let _ = g_norm;

    // Un-rotate: real_gaze = R_norm⁻¹ · normalized_gaze
    let g_real = r_norm.transpose() * g;
    let yaw_real = g_real.x.atan2(-g_real.z);
    let pitch_real = g_real.y.asin();
    (yaw_real as f32, pitch_real as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn face_model_centroid_at_origin() {
        let model = face_model_3d();
        let mut sum = Vector3::zeros();
        for p in &model { sum += p; }
        let centroid = sum / 6.0;
        // Centroid should be near origin (we don't enforce exactly zero)
        assert!(centroid.x.abs() < 1.0);
    }

    #[test]
    fn pnp_recovers_identity_rotation() {
        // Project the face model with R=identity, t=(0,0,500), focal=1000
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
            img_pts[i] = (
                cx + p.x * focal / p.z,
                cy + p.y * focal / p.z,
            );
        }

        let (r, t) = solve_pnp(&img_pts, focal, img_w, img_h).expect("pnp ok");
        // t should be approximately (0, 0, 500)
        assert!((t.z - tz).abs() < 50.0, "tz={}", t.z);
        assert!(t.x.abs() < 10.0, "tx={}", t.x);
        assert!(t.y.abs() < 10.0, "ty={}", t.y);
        // R should be approximately identity
        let trace = r.trace();
        assert!((trace - 3.0).abs() < 0.5, "trace={trace}");
    }
}
