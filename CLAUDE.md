# Saccade — Agent Briefing

Rust webcam eye tracker. Ridge regression on CLAHE eye patches → screen coordinates.
Goal: minimize pixel error on multi-point live validation.

## Current best results

| Metric | Value | Experiment | Config |
|--------|-------|------------|--------|
| Live (honest multi-point) | **237 px** | E12 | 5×5 grid, λ=auto, white BG |
| MPIIGaze (first-N protocol) | **5.31°** | E15 | 20×12, n_calib=500 |
| MPIIGaze (uniform-calib, n=200) | **3.85° / 2.97° median** | E16 | 20×12, uniform sampling |
| MPIIGaze (uniform-calib, n=500) | **3.53° / 2.73° median** | E16 | 20×12, uniform sampling |

Literature: L2CS-Net 3.92° (no calib), FAZE 3.18° (9-pt calib), GazeTR-Hybrid 3.43° (no calib).
**E16 key insight: calibration angle diversity beats feature engineering. Uniform sampling = -35% error.**

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
| Horizontal flip of right eye (E16) | 6.12° vs 5.89° — worse; MPIIGaze normalized space is already consistent | — |
| Sobel-x gradient features (E16) | 5.87° vs 5.89° at n=200 — negligible; no gain at n=500 | — |

## Promising next steps (ordered)

1. **Larger calibration grid in live app** — E16 shows calibration angle diversity is the key lever. 7×7 = 49 points covers more gaze angles than 5×5 = 25. Expected: significant improvement toward 3.85° level.
2. **n_calib=500 live session** — more clicks per point, or more points. E15 shows -10% on MPIIGaze first-N; E16 shows the uniform protocol already reaches 3.53°.
3. **Proper Sugano normalization** — fix `src/sugano.rs` iterative PnP. Expected: ~3.92° if matched correctly. 2-3 weeks.
4. **MediaPipe FaceMesh** — replace PFLD (68 pts) with 468-point model for better eye ROI.
5. **Accumulated clicks across sessions** — `saccade_calib.bin` already persists; just use it longer.

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
# Run MPIIGaze benchmark (45 sec) — standard first-N protocol
cargo run --release --example mpii_bench -- ./MPIIGaze_proc

# Best accuracy: uniform calibration sampling (simulates structured grid calib)
cargo run --release --example mpii_bench -- ./MPIIGaze_proc --n-calib 200 --uniform-calib
cargo run --release --example mpii_bench -- ./MPIIGaze_proc --n-calib 500 --uniform-calib

# Resolution ablation
cargo run --release --example mpii_bench -- ./MPIIGaze_proc --patch 30x18 --n-calib 500

# Query past results
grep '"mpii_deg"' results.jsonl | jq '{exp,patch,n_calib,variant,mean_deg}'
```
