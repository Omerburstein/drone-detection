# GLAD — architecture, training, and what is actually in the box

Reference notes on **GLAD** (Global-Local MAV Detection under Challenging Conditions
based on Appearance and Motion), Guo, Zheng, Zhang, Gao & Zhao, *IEEE T-ITS* 2024.

- Paper: [arXiv:2312.11008](https://arxiv.org/abs/2312.11008)
- Code: `github.com/WestlakeIntelligentRobotics/Global-Local-MAV-Detection`, MIT
- Local clone: [third_party/GLAD/](third_party/GLAD/) · yolov5 v6.0 at [third_party/yolov5/](third_party/yolov5/)

GLAD is the reference point for this project: it released the ARD-MAV dataset we test on,
our 15-video test split is its official split verbatim, and its per-condition numbers are
the only published figures our harness can be checked against.

> **Naming.** This is *GLAD*, not "GDAL" — GDAL is the unrelated geospatial raster library.

---

## 1. Architecture

Not one network. A **state machine over four detectors plus a classifier**, alternating
between a global (full-frame) and a local (cropped) regime.

```
                    ┌─ global regime (flag = 0) ───────────────────────┐
  frame ──────────▶ │  GAD: YOLOv5s on full 1920x1080, letterboxed 640 │
                    │        │ miss                                    │
                    │        ▼                                         │
                    │  GMD: motion detection on full frame             │
                    │        └── candidate ─▶ CNN classifier ─┐        │
                    └────────────────────────────────────────┼────────┘
                                        target acquired ─────┘
                                                 │
                    ┌─ local regime (flag = 1) ──▼─────────────────────┐
                    │  crop a 320x320 search region around last box    │
                    │  LAD: YOLOv5s-crop on the region                 │
                    │        │ miss                                    │
                    │        ▼                                         │
                    │  LMD: motion detection inside the region         │
                    │        └── candidate ─▶ CNN classifier           │
                    │  30 consecutive misses ─▶ back to global         │
                    └──────────────────────────────────────────────────┘
```

**Two separately trained YOLOv5s checkpoints**, not one model used twice — `yolov5s_GLAD.pt`
(full frame) and `yolov5s_GLAD-crop.pt` (320×320 crops). Verified locally: both are
`nc=1`, `names=['Drone']`, 7.02M params, strides 8/16/32.

**The motion module (`MOD2.py`)** is classical CV, no learning:

1. **Motion compensation** — grid of KLT points (96×72 px spacing global, 12×12 local),
   pyramidal Lucas-Kanade, RANSAC homography, warp the previous frame onto the current
   one. This is what makes frame differencing work from a *moving* camera.
2. **Segmentation** — absolute difference, adaptive threshold `T = 5 + mean(frameDiff)`,
   subtract the out-of-frame mask, median blur, morphological open then close ×3.
3. **Contour filtering** — keep blobs with `30 < area < 3000` px and aspect ratio 0.6–3.
4. **Motion classifier** — per candidate, Shi-Tomasi corners + LK inside the blob; reject
   if mean displacement is below 1 px, or if the std/mean ratio of displacement *or* angle
   exceeds 0.8. This is the "does it move coherently and differently from the background"
   test.
5. **Appearance classifier** — surviving crops resized to 32×32 into a small CNN; only
   class 1 (MAV) is accepted. This is the final gate against clutter.

**The classifier is LeNet-shaped**, not the deeper `MyNet` also defined in `Functions.py`:
2 conv (3→6→16, 5×5) + 2 maxpool + 3 FC (400→120→84→2). `MyNet` is dead code — the
248 KB of `Net_best.pth` matches `Net`'s ~61k parameters at fp32, so the shipped weights
are the LeNet.

---

## 2. How it was trained

| Component | Recipe |
| --- | --- |
| **GAD** — global YOLOv5s | 1920×1080 frames letterboxed to **640×640**, 150 epochs, batch 32, Adam, momentum 0.937, lr0 0.01 |
| **LAD** — local YOLOv5s | identical schedule, but training images are **320×320 crops centred on the target** |
| **CNN classifier** | 32×32 patches, 100 epochs, batch 64, lr 0.001, **46,268 clutter vs 17,695 MAV** |
| **Split** | ARD-MAV 45 videos train/val (random 5:1), 15 videos test — video-level, not frame-level |
| Also reported on | NPS-Drones (40/10 videos), Drone-vs-Bird (60/15) |

**The part worth copying:** the classifier's negatives were not random background crops.
They were mined from *the motion segmentation module's own intermediate output* — i.e. the
exact false positives the upstream stage actually produces. The 2.6:1 clutter-to-MAV ratio
is the empirical FP rate of stage 4. Any replacement classifier we train should be fed the
same way, or it will be strong on a distribution the pipeline never presents to it.

The 320×320 crop training for LAD is the second key idea: at 640×640 a 20 px target in a
1080p frame is ~6.7 px, below the stride-8 P3 cell — the checkpoint has **no P2 head**
(strides are 8/16/32), so the global detector structurally cannot resolve the small ones.
Cropping restores native scale. This is the same effect our `--tile` path exploits, and it
is why GAD-only recall is 0.17 while GAD+LAD is 0.51.

---

## 3. Published results

Test set = the 15 ARD-MAV videos in [MANIFEST.md](data/processed/ARD-MAV/MANIFEST.md).

**Per condition (full GLAD, Table III):**

| Condition | P | R | F1 | AP |
| --- | --- | --- | --- | --- |
| Ordinary | 0.99 | 0.96 | 0.97 | 0.91 |
| Complex background | 0.94 | 0.86 | 0.90 | 0.81 |
| **Small MAVs** | **0.82** | **0.67** | **0.73** | **0.58** |
| Total | 0.92 | 0.82 | 0.87 | 0.80 |

**Against baselines (Table IV, same split):**

| Method | P | R | F1 | AP | FPS |
| --- | --- | --- | --- | --- | --- |
| YOLOv5s-1 | 0.90 | 0.20 | 0.33 | 0.56 | 149.3 |
| YOLOv5s-2 | 0.78 | 0.41 | 0.54 | 0.61 | 88.5 |
| TPH-YOLOv5l-2 | 0.82 | 0.58 | 0.68 | 0.73 | 12.8 |
| Dogfight | 0.54 | 0.27 | 0.36 | 0.22 | 1.0 |
| MEGA | 0.45 | 0.35 | 0.39 | 0.31 | 3.5 |
| **GLAD** | **0.92** | **0.82** | **0.87** | **0.80** | **146.5** |

**Ablation (Table VII)** — this is the load-bearing table:

| Configuration | P | R | F1 | AP |
| --- | --- | --- | --- | --- |
| GAD only | 0.76 | 0.17 | 0.28 | 0.18 |
| GMD only | 0.81 | 0.30 | 0.43 | 0.25 |
| GAD+LAD | 0.90 | 0.51 | 0.65 | 0.54 |
| GMD+LMD | 0.89 | 0.30 | 0.45 | 0.34 |
| GAD+GMD+LAD | 0.90 | 0.78 | 0.84 | 0.72 |
| **GAD+GMD+LAD+LMD** | **0.91** | **0.81** | **0.86** | **0.80** |
| GLAD (full, + Kalman + adaptive region) | 0.92 | 0.82 | 0.87 | 0.80 |

Recall is where everything is won: 0.17 → 0.51 from the local region, 0.51 → 0.81 once the
motion branches are added. Precision barely moves (0.76 → 0.92). **The problem GLAD solves
is missed detections, not false alarms.**

**Speed** — 146.5 FPS average on an RTX 3070; 23.6 FPS on a Jetson Xavier NX. The average
is far above the module costs (GMD alone is 41.3 FPS on PC, 5.1 on Jetson) because the
expensive motion path only fires when appearance detection *fails*. On a hard sequence the
throughput approaches the GMD floor, so 23.6 FPS is a mean, not a guarantee.

---

## 4. What the authors say about it

**Stated limitations:**
- Accuracy degrades badly on very small MAVs (< 100 px area) — the 0.73 F1 / 0.58 AP row.
- **Hovering targets are missed** (Fig. 9a). After motion compensation a stationary MAV has
  near-zero residual motion, so the entire motion branch is blind to it — and the motion
  branch is what carries recall.
- **False positives on similar objects inside the local search region** (Fig. 9b). Once
  locked, the low confidence threshold and small region make it easy to latch onto a
  look-alike.

**Their future work, quoted:** *"our proposed algorithm will be extended to more types of
MAVs. Moreover, an end-to-end network is necessary to be designed to simplify the training
process, reduce the empirical parameters, and more effectively make use of the motion
clues."*

Read that as an admission: the pipeline has **three separate trainings and a large number
of hand-tuned constants**, and they know it is the weak point.

---

## 5. What is actually in the released repo

The README is explicit: *"This repository contains the basic codes for GLAD, the full codes
with Kalman Filter, Adaptive Search Region, and other codes will be published in the
future."* That is still true as of this survey.

**Consequences that matter for us:**

- **The released code is the `GAD+GMD+LAD+LMD` ablation row, not full GLAD.** `GLAD.py`
  uses a fixed `a = 160` (a constant 320×320 region); the paper's adaptive region is
  `L = 300 + 4·T_lost`. There is no Kalman filter anywhere in the clone. The honest
  reproduction target is therefore **P 0.91 / R 0.81 / F1 0.86 / AP 0.80**, and the
  per-condition rows in §3 are an *upper* bound.
- **The entry point needs TensorRT, but the PyTorch weights ship too.** `GLAD.py` loads
  `.engine` files via `detector*_trt.py` (TensorRT 7.2, `libmyplugins.so`, and a hardcoded
  `cuda.Device(1)` — the *second* GPU). None of that runs here. But `yolov5s_GLAD.pt` and
  `yolov5s_GLAD-crop.pt` are in the same folder and load cleanly on CPU through
  `third_party/yolov5` (v6.0, verified). **M4a is unblocked on this laptop** — it needs a
  detector shim swapping the three TRT classes for yolov5 CPU inference, not a GPU rental.
- **No training code.** Retraining means driving yolov5 v6.0 ourselves plus writing the
  classifier training loop and the crop/patch mining. The checkpoints do carry `optimizer`
  and `ema` state, so they are fine-tune-ready for M7.
- **No confidence scores leave the pipeline.** Output rows are `[frame, x, y, w, h]`.
  AP needs a ranking; a single-operating-point pipeline can only give P/R/F1 honestly.
  Score GLAD on P/R/F1 and treat any AP we compute for it as not comparable to ours.
- **Single target by construction.** `MOD2_global` `break`s on the first accepted candidate
  and returns one tuple; the state machine tracks one `init_rect`. It cannot report two
  drones. A hard architectural limit if the use case ever involves more than one intruder.

---

## 6. How to improve it

Ordered by (our expected gain) ÷ (effort). Items 1–3 are cheap and low-risk.

1. **Cache the classifier.** `Mynet_infer` constructs `Net()` and `torch.load()`s
   `Net_best.pth` from disk **on every candidate box, on every frame** — inside a loop that
   runs up to 50 times per frame ([Functions.py:464-484](third_party/GLAD/Functions.py#L464-L484)).
   Hoisting it to a module-level singleton is a few lines and should account for much of
   why GMD costs 41 FPS against LMD's 184. Batch the surviving crops while there.
2. **Fix the angle statistic.** `ratio_theta = std(theta)/mean(theta)` with `theta` in
   degrees from `arctan2` over (−180, 180]. The mean can pass through zero, making the
   ratio explode, and the wrap at ±180° gives a target moving left a huge spurious std —
   so **leftward motion is penalised**. Use circular statistics (mean resultant length)
   instead. This is a correctness bug in a rejection test, not a tuning choice.
3. **Make the constants scale-relative.** `area 30–3000`, `a = 160`, `dist_ref = 200`,
   blur kernel 11, `local_num == 30` are all absolute pixels tuned for 1920×1080.
   Anything at another resolution silently mis-filters. **This directly threatens M4b** —
   FL-Drones is not 1080p, so a naive run there would measure our failure to rescale, not
   GLAD's generalisation. Normalise by frame diagonal before that experiment.
4. **Attack the hovering failure.** The authors' own worst case. The appearance branch is
   the only thing that can see a stationary target, and it is the weak branch. Options:
   accumulate the difference over a longer baseline (frame *t* vs *t−k*) so slow relative
   motion integrates above threshold, or let the local regime fall back to appearance-only
   with a lowered threshold when motion returns nothing for several frames.
5. **Replace the LeNet gate.** It is the last thing standing between motion clutter and the
   output, and it is a 61k-parameter 1998-era network on 32×32 inputs. A small modern
   backbone at 64×64, trained on the same mined-negative distribution, is a cheap upgrade
   to the stage that determines precision. Keep the mining strategy — that is the good part.
6. **Add a P2 head to the global detector.** GAD's 0.17 recall is largely a stride problem
   (§2). A P2/stride-4 head, or simply running GAD tiled at native resolution the way our
   `--tile` path does, addresses the cause rather than patching around it with the local
   regime.
7. **Emit confidences and support N targets.** Needed before GLAD can be scored on the same
   footing as everything else in our ledger, and before it could handle a multi-drone
   scenario.

The authors' own prescription — one end-to-end network fusing motion and appearance — is
essentially **YOLOMG**, from the same lab lineage. That is the strategic argument for
treating GLAD as the baseline to beat rather than the thing to keep patching.

---

## 7. Comparability warnings

- **YOLOMG's 0.78 AP is not worse than GLAD's 0.80.** YOLOMG reports on **ARD100**
  (100 videos / 202,467 frames); GLAD reports on **ARD-MAV** (60 videos). Different
  datasets, different splits — the numbers do not belong in one table. YOLOMG-1280 reaches
  0.85 AP on ARD100 at 35 FPS, and 0.92/0.95 AP on NPS-Drones at 640/1280.
- **GLAD's ARD-MAV numbers are optimistic by construction.** Trained on the other 45 videos
  of the same capture campaign *and* architecturally tuned against this split. Reproducing
  them validates our harness (M4a). It does not tell us how GLAD generalises — that is M4b.
- The 640 vs 1280 spread inside YOLOMG's own results (0.78 → 0.85) is another instance of
  the resolution trap in CLAUDE.md: input size alone moves AP by 7 points.
