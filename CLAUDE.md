# Saccade — Agent Briefing

Rust webcam eye tracker. Ridge regression on CLAHE eye patches → screen coordinates.
Goal: minimize pixel error on multi-point live validation.

## Current best results

| Metric | Value | Experiment | Config |
|--------|-------|------------|--------|
| Live (honest multi-point) | **237 px** | E12 | 5×5 grid, λ=auto, white BG |
| MPIIGaze mean | **5.89° / 4.52° median** | E13 | 20×12 patches, n_calib=200 |
| MPIIGaze best n_calib=500 | **5.31°** | E15 | 20×12 patches, n_calib=500 |

Literature references: L2CS-Net 3.92° (no calib), FAZE 3.18° (9-pt calib), WebGazer ~4°.

## DO NOT RETRY — dead ends

| Approach | Why failed | Retry condition |
|----------|-----------|-----------------|
| CNN without Sugano normalization (E3–E8) | 300–737 px, worse than pixels | Only with iterative PnP + debug crops showing correct warp |
| Smooth pursuit calibration (E11) | 329 px vs 142 px — saccades + lag corrupt labels | Only with velocity-based saccade filtering + per-sample weights |
| Head pose features appended to pixel ridge (E2) | 256 px, worse — features washed out by λ | Only with separate regression head |
| ResNet-50 gaze CNN (E7) | <2 FPS — UI unusable | Only on hardware with GPU / NPU |
| Patch size > 20×12 at n_calib=200 (E15) | Plateau: 30×18→5.91°, 36×21→6.02°, no gain | Retry with n_calib≥500 |
| Zero-mean feature normalization (E12) | -3% improvement, not worth complexity | — |
| Decay sample weights (E12) | Neutral on this dataset, not worth complexity | Only for sessions with strong temporal drift |

## Promising next steps (ordered)

1. **n_calib=500 live session** — free win; E15 shows -10% on MPIIGaze. Just run a 500-click calibration.
2. **Proper Sugano normalization** — fix `src/sugano.rs` iterative PnP, add visual debug of warped crop, match to ResNet-18/L2CS-Net training params. Expected: ~3.92° if matched correctly. 2-3 weeks.
3. **MediaPipe FaceMesh** — replace PFLD (68 pts) with 468-point model for better eye ROI. Significant for any geometric approach.
4. **Accumulated clicks across sessions** — `saccade_calib.bin` already persists; just use it longer.

## Key files

| File | Purpose |
|------|---------|
| `examples/webgazer.rs` | Main live app (pixel ridge) |
| `examples/webgazer_cnn.rs` | CNN variant with Sugano normalization |
| `examples/mpii_bench.rs` | MPIIGaze benchmark (`--patch WxH --n-calib N`) |
| `src/ridge.rs` | Ridge regressor + CLAHE feature extraction |
| `src/sugano.rs` | ETH-XGaze normalization (working but PnP is approximate) |
| `tools/preprocess_mpii.py` | MPIIGaze .mat → PNG+TSV converter |
| `EXPERIMENTS.md` | Full experiment log (read for details) |
| `results.jsonl` | Machine-readable benchmark numbers (one line/run) |

## Architecture

```
camera → rustface (face bbox) → PFLD (68 landmarks) → eye crop
→ CLAHE + bilinear resize to 20×12 → 480-D feature vector (×2 eyes + 3 head)
→ RidgeRegressor (LOO CV λ-tuning, n=45 clicks) → screen (x, y)
→ OneEuroFilter → cursor
```

## Benchmark commands

```sh
# Run MPIIGaze benchmark (45 sec)
cargo run --release --example mpii_bench -- ./MPIIGaze_proc

# Resolution ablation
cargo run --release --example mpii_bench -- ./MPIIGaze_proc --patch 30x18 --n-calib 500

# Query past results
grep '"mpii_deg"' results.jsonl | jq '{exp,patch,n_calib,mean_deg}'
```
