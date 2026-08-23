# Edge budget — can the detector run where it has to run?

Counterpart of [experiments.md](experiments.md). The ledger says **how well**; this says
**whether it can run, on what, and at what cost.** Maintained by `deploy-agent`. Accuracy
numbers remain `algo-agent`'s property, including the ones a board produces.

> ## ⚠️ Almost everything below the "Where the time goes" section is ASSUMPTION.
>
> **No target hardware exists yet.** Not one number in this document was measured on an
> edge board. The measured columns are all from this laptop (`i7-1255U`, CPU, fp32
> PyTorch) or from published papers under *their* conditions, not ours. Every projection
> onto a Jetson is arithmetic on somebody else's benchmark and is marked
> **[ASSUMED]** or **[EXTRAPOLATED]**. See [§8](#8-assumed-vs-measured) for the full
> separation.
>
> **Two inputs are the user's to supply and nothing can be sized without them:**
> assumed **closing speed** and required **frames of persistence**. [§1](#1-the-requirement)
> works an illustration with invented values; it is an illustration, not a budget.

---

## 0. Verdict

**The current detector is not slow. The host is.** Swapping the model is the wrong lever
and would cost most of what the project has built.

Three numbers settle it, all from this repo:

| | fps on this laptop | recall (centre@1×) | tiny-target recall |
| --- | --- | --- | --- |
| `yolov8n` @720p whole-frame | **11.0** | not scored — the `yolov8s` runs below bound it | — |
| `yolov8s` @640 whole-frame (EXP-001) | 3.24 | 0.0121 | **0.0000** |
| `yolov8s` @1280 whole-frame (EXP-002) | 1.33 | 0.0550 | **0.0000** |
| `yolov8s` tiled ×8 (EXP-003) | 0.64 | 0.1246 | 0.0289 |
| **GLAD** (EXP-004) | **2.67** | **0.8946** | **0.8493** |

**GLAD is the second-fastest configuration this project has measured and carries 7–70× the
recall of every one of them.** The faster options are faster precisely because they are
not doing the work — EXP-001–003 established that appearance-only detection at 10–30 px
fails outright, and the two fastest configurations score exactly zero on tiny targets.
There is no speed/accuracy trade to make here; the fast end of the curve is empty.

The 2.67 fps is a **CPU-only laptop with no CUDA** running fp32 PyTorch. The same pipeline
is published at **23.6 FPS on a 2019 Jetson Xavier NX** and 146.5 FPS on an RTX 3070
([glad-model.md §3](glad-model.md)). That is **~8.8× and ~55×** — the entire gap is the
host, and it is closed by a **$249 purchase**, not by a research programme.

**Recommendation in one line:** buy a Jetson Orin Nano Super 8 GB, port GLAD's two
`yolov5s` checkpoints to TensorRT 10 FP16 on the board, keep the architecture, and spend
the saved effort on the validation run that proves the port did not break it.
Full form in [§6](#6-recommendation).

**The caveat that governs every accuracy statement below:** the moment a model is swapped,
re-exported or quantised, **every accuracy number in this document and in the ledger stops
applying to it.** Nothing here may be cited as the accuracy of a swapped or quantised
detector until it has been re-scored through `src.evaluate` on this project's split at the
same match criterion — and that verdict belongs to `algo-agent`, not to this document.
See [§7](#7-export-and-quantisation-is-a-protocol-change).

---

## 1. The requirement

**Derived, never inherited.** "30 FPS" is not a requirement, it is a familiar number.

### The two unknowns

| Input | Status | Who decides |
| --- | --- | --- |
| **Closing speed** | ⚠️ **UNKNOWN — not measured, not specified** | the user |
| **Persistence frames** before the system acts | ⚠️ **UNKNOWN — not measured, not specified** | the user |

Everything else in this section is downstream of those two. Until they are supplied, no
board can be *justified* — only, as in [§6](#6-recommendation), recommended on the
separate ground that it is cheap and strictly dominates the alternatives.

### What the budget actually is

**Closing speed × end-to-end latency = how much nearer the target is before anything
reacts.** That distance is the thing being budgeted, not the frame rate.

**End-to-end means capture → decode → detect → track → decide.** The model forward pass is
one term. Note that Ultralytics' Jetson benchmark tables — the ones quoted in
[§5](#5-alternatives) — state explicitly that *"inference time does not include
pre/post-processing"*. A 5 ms inference behind a 100 ms capture-and-decode chain is a
105 ms system, and camera buffering can hide another frame or two on top.

**Detection latency is not inference latency.** If the design needs N frames of persistence
before it will act, true latency is `N / FPS + pipeline`. This is where frame rate buys
back real time, and it is the reason the worst-case FPS matters far more than the mean.

### Illustration — invented numbers, do not cite

**[ASSUMED]** Two multirotors, 20 m/s each, head-on ⇒ **40 m/s closing**. Persistence
N = 3 frames. Pipeline overhead (capture + decode + track + decide) 60 ms.

| Scenario | detect FPS | latency = 3/FPS + 60 ms | target closes |
| --- | --- | --- | --- |
| Orin Nano, easy scene **[EXTRAPOLATED]** | 60 | 110 ms | 4.4 m |
| Orin Nano, hard scene (motion path firing) **[EXTRAPOLATED]** | 14 | 274 ms | **11.0 m** |
| This laptop, sustained (EXP-004, measured) | 2.67 | 1,184 ms | **47.4 m** |

**The hard-scene row is the budget.** A pipeline that keeps up over empty sky and drops
over clutter has failed, because clutter is the only moment detection mattered.

### The resolution floor, and why it is not negotiable

**[ASSUMED]** 0.3 m airframe, 1920 px wide sensor, ~60° horizontal FOV ⇒ 0.031°/px.

| Apparent size | Range | Time to contact at 40 m/s |
| --- | --- | --- |
| 10 px (first detectable) | ~55 m | **1.4 s** |
| 30 px | ~18 m | 0.5 s |

Sanity check against real data: ARD-MAV's own p95 closest approach is **34.2 px**
([experiments.md](experiments.md)), i.e. ~16 m under these assumptions, and its smallest
band is <8 px, ~70 m. The dataset's range span and this arithmetic agree.

**The whole engagement is on the order of one second.** That is what makes the latency
budget tight, and it is why **input resolution cannot be traded for frame rate**: at 55 m
the target is 10 px in a 1920-wide frame. Halving the input to 960 makes it 5 px and the
detection range collapses. This project's founding trap
([CLAUDE.md](../CLAUDE.md)) wearing a deployment badge.

**Resolution that must be fed: 1920×1080, full rate, contiguous.** Contiguous is not
stylistic — GLAD's motion branches difference against the previous frame, so `src.glad_detect`
has no `--stride` by deliberate design. **Frame striding is not available as a speed lever
on this pipeline.**

---

## 2. Where the time actually goes — the decomposition

This is the section with real measurements in it.

### 2.1 Everything measured on this machine

All from this repo. Host is `i7-1255U`, 16 GB, no CUDA, `torch 2.9.1+cpu`, fp32.

| Source | Configuration | Frames | Wall clock | fps | ms/frame |
| --- | --- | --- | --- | --- | --- |
| EXP-001 | `yolov8s`, 1080p → 640 letterbox, whole-frame | 2,834 | 875 s | 3.24 | 309 |
| EXP-002 | `yolov8s`, 1080p → 1280 letterbox | 2,834 | 2,138 s | 1.33 | 754 |
| EXP-003 | `yolov8s`, 8 × 640 native tiles | 2,834 | ~4,400 s | 0.64 | 1,552 |
| **EXP-004** | **GLAD, 1080p, full pipeline, sustained** | **28,337** | **10,602 s** | **2.67** | **374** |
| [baseline_detect.md](baseline_detect.md) | `yolov8n` @720p whole-frame | — | — | 11.0 | 91 |
| [baseline_detect.md](baseline_detect.md) | `yolov8n` @720p tiled | — | — | 0.72 | 1,389 |
| [glad_detect.md](glad_detect.md) | GLAD, spot figure | — | — | ~4.8 | 208 |

> **First correction to a premise.** The **11 fps / 0.72 fps** pair quoted in
> [CLAUDE.md](../CLAUDE.md) is `src.baseline_detect` with **`yolov8n` at 720p** — a
> different CLI, a different model and a different resolution from GLAD. It is not GLAD's
> number and must not be read as one. It is also the fastest thing this project can run
> *and* among the least accurate.
>
> **Second correction.** GLAD's own two figures disagree: **~4.8 fps** (spot, quoted in
> [glad_detect.md](glad_detect.md)) against **2.67 fps** (sustained, EXP-004, 28,337
> frames / 2.9 h). The sustained figure is the one to budget against — this is the
> mean-vs-worst-case rule applying to our own measurement, not just to vendors'. A third
> observation taken while writing this document (`exp005`, ARD100, in progress) read
> 3.16 → 3.92 fps on `phantom03`. **Throughput on this pipeline spans roughly 2.7–4.8 fps
> depending on content.** Any single figure quoted without its content is incomplete.

### 2.2 Model, host, or protocol?

**Protocol multipliers — measured, model held constant:**

| Lever | Cost multiplier | Source |
| --- | --- | --- |
| 640 → 1280 letterbox, whole-frame | **2.44×** | EXP-001 → EXP-002 |
| whole-frame → 8 native tiles | **5.02×** | EXP-001 → EXP-003 |
| whole-frame → tiled, `yolov8n` @720p | **15.3×** | baseline_detect.md |

The 5× and 15× are both real; the multiplier depends on tile count and on how much fixed
decode/IO overhead the model's own pass hides. **Budget tiling at 5–15×, not at "tile
count ×".**

**And then: none of it applies to GLAD.** GLAD has no tiling switch and no resize switch —
[experiments.md](experiments.md) says so explicitly, and it is architectural, not a missing
flag. Its global 640 letterbox and its 320×320 local crop are *states of one state machine*.
So the protocol contribution to GLAD's cost is **~1.0×. There is nothing to claw back
here.**

**Host multiplier — published, same pipeline:**

| Host | FPS | vs our 2.67 |
| --- | --- | --- |
| i7-1255U, CPU, fp32 PyTorch (ours, EXP-004) | 2.67 | 1× |
| Jetson Xavier NX, TensorRT (GLAD paper) | 23.6 (mean) | **8.8×** |
| RTX 3070, TensorRT (GLAD paper) | 146.5 (mean) | **55×** |

**Model multiplier: ~1.0×.** GLAD's steady state is one `yolov5s` forward pass per frame
(see 2.3). `yolov5s` at 640 is 7.02M params / ~16.5 GFLOPs — already near the cheapest
architecture that does this task at all. The models that are meaningfully cheaper are the
ones EXP-001–003 measured at 0.006–0.025 AP.

> ### The decomposition, stated
>
> | Cause | Contribution to the 2.67 fps |
> | --- | --- |
> | **Host** — CPU-only, no CUDA, fp32 PyTorch | **~9× against a 2019 edge board; ~55× against a desktop GPU** |
> | **Protocol** — tiling / resolution | **~1× — GLAD neither tiles nor over-resizes; nothing available** |
> | **Model** — architecture cost | **~1× — already the cheap end of what works** |
>
> **The slowness is the host, essentially in full.**

### 2.3 Inside GLAD — why the state machine is not the problem

Branch distribution from EXP-004, 28,337 frames:

| Branch | Share | What runs |
| --- | --- | --- |
| `local yolo` | **88.4%** | LAD only — one `yolov5s` pass on a 320×320 crop upscaled to 640. **No motion module at all.** |
| `local miss` | 7.4% | LAD + LMD + classifier |
| `global miss` | 2.9% | GAD + GMD (+ classifier) |
| `local mod` | 1.0% | LAD + LMD + classifier |
| `global mod` / `global yolo` | 0.2% | — |

**The modal frame — 88.4% of them — is one small YOLO forward pass.** The expensive motion
path (KLT grid, pyramidal Lucas-Kanade, RANSAC homography, full-frame warp, median blur,
3× morphological open/close over 1920×1080) fires on **11.6%** of frames. Acquisition
happens 42 times in 28,337 frames.

That local search region is not overhead — **it is the optimisation.** Without it every
frame would pay a full-frame pass; with it 88% of frames pay a crop-sized one. The
"tracking between detections" idea is already implemented and already banked.

**[ESTIMATED — not measured]** Solving `0.884·T_LAD + 0.116·(T_LAD + T_MOD) = 374 ms`
with `T_LAD ≈ 200 ms` (scaling EXP-001's 309 ms by the yolov5s/yolov8s FLOP ratio) gives
`T_MOD ≈ 1.5 s` per motion invocation. Directionally consistent with the paper's own
module split — GMD alone is 41.3 FPS on a 3070 against 146.5 for the whole pipeline, and
5.1 FPS on Xavier NX against 23.6.

> **This estimate is arithmetic, not a profile.** A real per-stage timing run is the
> single most valuable measurement this document is missing, and it is cheap: a
> `time.perf_counter()` around the four stages in `src/algo/glad/pipeline.py` over ~2,000
> contiguous frames. **It could not be taken while writing this** — `exp005` was
> occupying the CPU, and a contended measurement violates the charter's own benchmarking
> rules. Logged in [todo.md](todo.md).

---

## 3. The cheap wins, before any swap

Ordered by (expected gain) ÷ (risk). **Every one of them requires an accuracy re-score
before its number may be carried forward** — see [§7](#7-export-and-quantisation-is-a-protocol-change).

| # | Lever | Expected speedup | Accuracy caveat |
| --- | --- | --- | --- |
| 1 | **ONNX Runtime / OpenVINO CPU export** of the two `yolov5s` checkpoints | **1.5–3×** on the YOLO stage, which is ~88% of frames' dominant cost. Ultralytics claims "up to 3× on CPU" | fp32→fp32 is numerically close but **not identical** — the NMS implementation changes. Re-score. Low risk. |
| 2 | **OpenVINO on the Iris Xe iGPU** | Unquantified for this exact chip. One third-party anchor: >50 FPS for YOLOv8 INT8 on an i7-12700H iGPU | Same as (1), plus INT8 if used. **Local convenience only — the Iris Xe is not an edge stand-in** (charter). Does not help the target board. |
| 3 | **TensorRT FP16** *(on-board only)* | **1.65×** over TensorRT FP32, measured by Ultralytics on Orin Nano Super (YOLO26n 7.53 → 4.57 ms) | Usually within noise; on their table mAP50-95 0.477 → 0.480. **Still re-score** — small targets are the fragile case. |
| 4 | **TensorRT INT8** *(on-board only)* | **1.20×** further (4.57 → 3.80 ms) | ⚠️ **Poor value here.** Same table: mAP50-95 **0.480 → 0.449**, and the literature reports 3–7 points absolute mAP50-95 loss with *smaller models worse*. 1.2× for the project's most fragile property. **Do not take this until FP16 has been proven on the size cut.** |
| 5 | **PyTorch → TensorRT** *(on-board only)* | **2.07×** at equal FP32 precision (15.60 → 7.53 ms, same table) | Pure runtime change at equal precision. Re-score anyway; the port is new code. |
| 6 | **Tile-grid / resolution choices** | **None available.** | GLAD has no tile or resize switch. Lowering input re-creates the founding trap. The only resolution lever points *upward* (GAD tiled, or a P2 head — [glad-model.md §6](glad-model.md) item 7), costing speed to buy acquisition. |
| 7 | **Frame striding** | **Not available.** | Incoherent for GLAD: both motion branches difference consecutive frames. `src.glad_detect` has no `--stride` on purpose. A strided sample measures a different algorithm. |
| 8 | **Tracking between detections** | **Already banked.** | This *is* the local regime. 88.4% of frames already run a crop-sized pass instead of a full-frame one. |
| 9 | **Cache the classifier** | **Already banked.** | Upstream reloaded `Net_best.pth` from disk per candidate per frame, inside a loop running up to 50× per frame. Our port hoists it (`src/algo/glad/classifier.py`, `load_gate`). [glad-model.md §6](glad-model.md) item 1 attributes much of the GMD/LMD cost gap to this. |
| 10 | **Run LAD at native 320 instead of upscaling to 640** | **~4× FLOPs on the modal frame** — the largest single win identified | ⚠️ **This is a retrain, not a config change.** `yolov5s_GLAD-crop.pt` was trained on 320×320 crops fed at 640. Changing the input scale changes the detector. **`algo-agent`'s call**, candidate for M7. |

**Realistic stack on the target board: (5) × (3) ≈ 3.4× over on-board PyTorch FP32**,
before any architectural change, at a re-scoring cost of two `src.evaluate` runs over a
persisted JSONL.

---

## 4. Board selection

**Baseline: Jetson Orin Nano Super Developer Kit, 8 GB.**

| | Orin Nano Super 8 GB | Xavier NX (the published GLAD board) |
| --- | --- | --- |
| Price | **~$249** (down from $499) | ~$399, EOL-adjacent |
| Compute | 67 TOPS INT8 (sparse) | 21 TOPS INT8 |
| Memory bandwidth | **102 GB/s** (LPDDR5) | 51 GB/s (LPDDR4x) |
| Memory | 8 GB shared | 8 GB shared |
| Power modes | 7 W / 15 W / 25 W MAX | 10 W / 15 W |
| Toolchain | JetPack 6.x, **TensorRT 10.x** | JetPack 4.x/5.x, TensorRT 7.x/8.x |

**Nominal ratio ~3.2× on TOPS and 2× on bandwidth.** For this workload — small batch,
small model, high input resolution — **bandwidth is likelier to bind than TOPS**, so
2× is the more honest scaling figure than 3.2×.

**Name the power mode.** An Orin Nano at 7 W and at 25 W are effectively different boards.
Every projection in this document **[ASSUMES] 15 W**, which is the middle mode and the one
a small airframe is likeliest to sustain thermally. Ultralytics' benchmark table
**[ASSUMES] MAX power mode with Jetson Clocks**, so their millisecond figures are an
optimistic bound for us, not a match.

### ⚠️ Toolchain break — load-bearing, and it is not an export

GLAD's released entry point deserialises **TensorRT 7.2** engines through `detector*_trt.py`,
with `libmyplugins.so` and a hardcoded `cuda.Device(1)`. Confirmed by search: **TensorRT
engines are architecture- and version-specific and do not load across JetPack versions**;
anything built against JetPack 4's toolchain must be *rebuilt*, not recompiled, and the
TRT 8 → 10 API changed enough that conversion scripts need updating. The tensorrtx plugin
`.so` would need rebuilding against the TRT 10 API.

Combined with the charter's standing warning — **every number in our ledger, EXP-004
included, came from `src/algo/glad/`, our CPU `yolov5` shim, not from TensorRT** — the
conclusion is:

> **Deploying GLAD to a Jetson is a new implementation requiring its own validation run.
> It is not an export.** Budget it as such: expect to go `.pt` → ONNX → TensorRT 10 FP16
> on the board, rebuilding the post-processing rather than porting `detector*_trt.py`, and
> then to re-run the full EXP-004 split through `src.evaluate` and compare side by side.

**Memory, not compute, is the plausible binding constraint** at 1920×1080 with a 640
global network plus a 640 local network plus full-frame OpenCV motion buffers on a shared
8 GB pool. Unmeasured. If it binds, the escape is Orin NX 16 GB, not a smaller model.

### 4.1 Orin NX 16 GB — what the step up actually buys

Asked directly, and the answer is smaller than the price gap suggests.

| | Orin Nano Super 8 GB | **Orin NX 16 GB** | ratio |
| --- | --- | --- | --- |
| CUDA / tensor cores | 1024 / 32 | 1024 / 32 | **1.00×** |
| Memory bandwidth | 102 GB/s LPDDR5 | 102.4 GB/s LPDDR5 | **1.00×** |
| CPU | 6× A78AE @1.7 GHz | 8× A78AE @2.0 GHz | ~1.6× |
| Peak INT8 | 67 TOPS | 100 TOPS | 1.5× (tensor-core peak; not reached here) |
| Memory | 8 GB shared | **16 GB shared** | 2× |
| Power modes | 7 / 15 / 25 W | 10 / 25 W (40 W Super) | — |
| Price | ~$249 | ~$600–900 | **2.5–3.5×** |

**The GPU is the same silicon at a slightly higher clock, on the same memory bus.** Since
§4 argues this workload is bandwidth-bound rather than TOPS-bound, the 1.5× TOPS figure
does not transfer, and the empirical anchor confirms it: Ultralytics benchmark **YOLO26n,
TensorRT FP16, 640, batch 1, MAX power, pre/post excluded** at **4.13 ms on Orin NX 16 GB
against 4.57 ms on Orin Nano Super — 1.11×**. That is GLAD's modal frame almost exactly
(88.4% of frames are one small `yolov5s` pass), so it is the right multiplier for it.

The CPU is where Orin NX is genuinely ahead, and GLAD's expensive branch — KLT grid,
pyramidal Lucas–Kanade, RANSAC homography, full-frame warp, morphology — is OpenCV on the
CPU. Two more cores at a higher clock is ~1.6× on paper, less in practice because LK and
RANSAC do not scale linearly.

#### GLAD projection **[EXTRAPOLATED — no board has been benchmarked]**

Built by scaling the paper's published Xavier NX figures the same way §4 does, then
applying 1.11× (GPU path) and ~1.3–1.5× (motion path):

| Branch | Xavier NX (published) | Orin Nano Super | **Orin NX 16 GB** |
| --- | --- | --- | --- |
| Modal frame, LAD only (88.4%) | ~30 | ~60 | **~65** |
| Motion path firing (11.6%) | **5.1** (GMD, published) | ~10–14 | **~14–19** |
| Sustained mean over content | 23.6 (published mean) | ~47 | **~50–55** |

> **Answer in one line: roughly 50 FPS mean and — the number that matters — 15–20 FPS in
> the hard scenes, about 1.15× an Orin Nano Super for 2.5–3.5× the money.**
>
> Every figure above is arithmetic on someone else's benchmark, at **MAX power**, with
> pre/post-processing excluded, and through a TensorRT path **this project has never run**
> (§4's toolchain break). Treat the shape — modal frame cheap, motion path 4–5× worse —
> as the finding; treat the digits as provisional until a board is in hand.

**So buy Orin NX for the 16 GB, not for the frame rate.** §4 already names memory as the
plausible binding constraint at 1080p; that is the case for this board, and it is a good
one. Frame rate is not.

**What would change the board recommendation** — see [§6](#6-recommendation).

---

## 5. Alternatives

Every published figure is annotated with what it was measured under. **A figure whose
board, precision, resolution, batch size and content are not all ours is
non-transferable** and is marked so.

| Model | Published speed | Board / precision / res / batch / content | Accuracy | Transferable? |
| --- | --- | --- | --- | --- |
| **GLAD** (incumbent) | 146.5 FPS / **23.6 FPS** | RTX 3070 / Xavier NX · TensorRT · 1080p · batch 1 · **mean over ARD-MAV** | Ours, EXP-004: P .856 / R .771 / F1 .811 @IoU 0.50; P .993 / R .895 @centre | ⚠️ **Mean only.** GMD alone is 5.1 FPS on the same NX — **throughput collapses ~4.6× exactly when the scene is hard.** Budget the floor, not the mean. |
| **YOLOMG-640** | 133 FPS | **RTX 2080Ti** · precision unstated · 640 · batch unstated · ARD100 | AP@0.5 **0.78** on ARD100 | ❌ **Non-transferable.** A 2080Ti is ~4–8× an Orin Nano. Different dataset *and* split from EXP-004 — per [CLAUDE.md](../CLAUDE.md) the numbers do not belong in one table. |
| **YOLOMG-1280** | 35 FPS | as above, 1280 | AP@0.5 **0.85** on ARD100 | ❌ Same. Note the 640→1280 spread (0.78→0.85) is 7 AP points **from resolution alone**. |
| **A2A-YOLO** (Remote Sensing, 2026) | **15 FPS** | **RK3588** NPU · precision/res unstated · air-to-air | Det-Fly AP **0.819** | ❌ **Non-transferable and wrong size regime.** Det-Fly is 4K single-Mavic with large targets. Appearance-only, so EXP-001–003's finding applies: appearance alone fails at 10–30 px. |
| **TransVisDrone** | ~31 FPS reported for a Swin-T backbone at 640 on Xavier NX (attribution loose — verify before citing) | Xavier NX · 640 | AP@0.5: NPS 0.95, FL-Drones 0.75, AOT 0.80 | ⚠️ Spatio-temporal, so it uses motion correctly. Memory-hungry; 8 GB may bind at 1080p. **Unverified latency.** |
| **YOLO26n / YOLO11n / YOLOv8n** | **3.80 ms/im = 263 FPS** | **Orin Nano Super** · TensorRT **INT8** · **640** · batch 1 · COCO · **MAX power** · *excludes pre/post-processing* | COCO mAP50-95 0.449 | ❌ **This is the "faster model" the question reaches for, and this project has already measured what it is worth:** EXP-001–003, AP@0.5 **0.006–0.025**, tiny-target recall **0.000**. Speed was never the constraint. |
| **Dogfight** | 1.0 FPS | GLAD paper's own comparison table | AP 0.22 (ARD-MAV) | ❌ Slower and far worse. |

### What each would cost in small-target recall — honestly

- **GLAD → any plain YOLO (n-class):** catastrophic. Measured, not estimated: tiny-target
  recall 0.849 → **0.000–0.029**. This is not a trade, it is a removal of the capability.
- **GLAD → YOLOMG:** genuinely plausible as an *accuracy* move, and it is the authors' own
  prescription (end-to-end motion+appearance fusion). But it has **no pretrained weights**,
  is **GPL-3.0** (flag if commercial), and its ARD100 numbers are not comparable to our
  ARD-MAV ones. **Cost: a GPU training run, and an unknown small-target result on our
  split until measured.** It also removes GLAD's single-target architectural limit, which
  matters if the use case is ever multi-intruder.
- **GLAD → A2A-YOLO / RK3588 class:** buys a lower power envelope. Costs the motion branch,
  which is what carries GLAD's recall from 0.51 to 0.81 in the paper's own ablation.
  Expect a large small-target loss. Unquantified on our split.
- **GLAD FP32 → FP16:** expected near-zero, unproven. **→ INT8:** expect 3–7 points
  absolute, **concentrated in the smallest size bands** — which for this project is where
  the target population actually is (18,265 of ARD-MAV's 28,160 targets are <16 px).

---

## 6. Recommendation

**Run GLAD, architecturally unchanged, on a Jetson Orin Nano Super 8 GB (~$249) at 15 W.**

1. **Buy the board.** $249 against ~9× throughput is the cheapest measured improvement
   available to this project by a wide margin. No model swap comes close on either axis.
2. **Port, do not export.** `.pt` → ONNX → TensorRT 10 **FP16**, built on the board.
   Rebuild the post-processing; do not attempt to revive `detector*_trt.py` (TRT 7.2,
   dead toolchain, hardcoded second GPU). Leave `MOD2.py`'s motion module on the CPU/OpenCV
   path initially — it is 11.6% of frames and porting it is a separate project.
3. **Stop at FP16.** INT8 buys 1.2× and risks the smallest size band, which is the whole
   problem. Revisit only if the derived latency budget genuinely demands it.
4. **Validate before believing.** Re-run the full EXP-004 split through the board build,
   score with `src.evaluate` at the *same* criterion, and put both columns side by side.
   Then re-cut by size with `src.plot_eval` / `src.cross_eval` — that evidence is a re-cut
   of a persisted dump, not a new experiment. **The verdict on whether accuracy survived is
   `algo-agent`'s to give.**
5. **Benchmark properly on arrival.** Warm up, then sustained load over contiguous video —
   not a frame sample. Report **p50 / p95 / p99 and the worst sequence**, per stage, with
   the power mode named. Expect the worst sequence to be `phantom63` or `phantom43`: they
   already carry most of EXP-004's accuracy loss and, being the frames where appearance
   fails, they are also where the expensive motion path fires.
6. **Do the cheap local win in parallel** — ONNX/OpenVINO CPU export (§3 item 1) makes
   every future local run 1.5–3× faster for a few hours of work, and the exported artifact
   is the same ONNX the board build starts from.

### What would change my mind

| Trigger | New recommendation |
| --- | --- |
| Derived requirement needs **>60 FPS end-to-end at 1080p** worst-case | **Orin NX 16 GB**, not a smaller model — the resolution floor is not negotiable |
| Memory binds at 1080p on 8 GB | **Orin NX 16 GB**. Memory, not compute, is the likely constraint |
| **Multi-intruder** use case | GLAD is **architecturally disqualified** — `MOD2_global` breaks on the first candidate and the machine tracks one box. **YOLOMG becomes the baseline regardless of speed.** |
| **M4b (`exp005`, running now) shows GLAD's accuracy does not survive unseen video** | The swap question reopens — but as an **accuracy** question, not a speed one, and it is `algo-agent`'s |
| Sustained power budget **below ~10 W** | RK3588 / A2A-YOLO class becomes relevant, at a large and currently unquantified small-target cost |
| FP16 re-score shows loss in the smallest size band | Drop to **TensorRT FP32** — still ~2× over on-board PyTorch — rather than chasing INT8 |
| Commercial use case | Flag licensing: GLAD is **MIT** (fine); **YOLOMG is GPL-3.0**; Drone-vs-Bird data is DUA-gated and non-commercial |

---

## 7. Export and quantisation is a protocol change

`algo-agent` enforces that runs at different `imgsz` or thresholds are not comparable.
**The same rule applies to precision and to the export toolchain**, and it is easier to
violate because an export feels like packaging.

- **FP32 → FP16 → INT8 changes the detector.** Never carry an accuracy number across an
  export. Re-score the exported artifact through `src.evaluate` on the same split, the same
  criterion and the same persisted-JSONL path, and publish **both** numbers side by side.
- **Small targets are the fragile case.** A 10–30 px drone is a handful of activations;
  quantisation error invisible on a 200 px object can erase it. Detection heads at
  different scales have measurably different sensitivity to quantisation perturbation.
  **Expect the loss in the smallest bands and check there specifically** — `src.plot_eval`
  and `src.cross_eval` already cut by size, so the evidence is a re-cut, not a new run.
  The band to watch is `<8 px` and `8–12 px`, which is where EXP-004's localisation error
  is already 0.255 and 0.176 target-widths.
- **Never buy frame rate with input resolution without saying so.** See [§1](#1-the-requirement).
- **INT8 calibration is data.** It needs a representative sample — same conditions, same
  target sizes, and **never the test split**. Coordinate with `dataset-agent`; ARD-MAV's
  45 training videos are the right source.
- **The metric to watch when any future build claims a small-target win** is `loc_err`
  by size, not aggregate P/R. EXP-004 established that relative localisation error degrades
  5× monotonically as targets shrink, and that this — not the detector — is what moves P/R
  by 23 points between IoU 0.40 and 0.50.

**Required ledger field, from M6 onward:** every entry that produces numbers on target
hardware records **board, power mode, precision, input resolution, batch size, and
sustained fps with p95** alongside its accuracy. A mean fps with no content description is
an incomplete entry.

---

## 8. Assumed vs measured

### Measured, and reliable

| Fact | Source |
| --- | --- |
| GLAD sustained 2.67 fps over 28,337 frames on `i7-1255U` CPU fp32 | EXP-004 |
| GLAD spot 4.8 fps; observed 3.16–3.92 fps on ARD100 `phantom03` | glad_detect.md; `exp005` log |
| Baseline `yolov8s` 3.24 / 1.33 / 0.64 fps at 640 / 1280 / tiled | EXP-001–003 |
| `yolov8n` 11.0 / 0.72 fps at 720p whole-frame / tiled | baseline_detect.md |
| Tiling costs 5.0× (`yolov8s` @640) to 15.3× (`yolov8n` @720p) | EXP-001→003; baseline_detect.md |
| 640 → 1280 costs 2.44× | EXP-001 → EXP-002 |
| GLAD's branch split: 88.4% `local yolo`, 11.6% motion path | EXP-004 |
| Recall/precision for every configuration above | experiments.md |
| GLAD has no tile or resize switch | experiments.md, glad_detect.md — architectural |
| GLAD is single-target by construction | glad-model.md §5, pipeline.py |
| Machine is CPU-only, no CUDA, no XPU | hardware.md |

### Published by others under *their* conditions — not ours

| Claim | Condition | Status |
| --- | --- | --- |
| GLAD 23.6 FPS Xavier NX / 146.5 FPS RTX 3070 | TensorRT, batch 1, **mean over content**; GMD floor 5.1 / 41.3 FPS | ⚠️ mean, not guarantee |
| YOLOMG 133 / 35 FPS @640 / @1280 | **RTX 2080Ti**, precision & batch unstated | ❌ non-transferable to edge |
| A2A-YOLO 15 FPS | RK3588, precision & resolution unstated, Det-Fly (4K, large targets) | ❌ non-transferable |
| YOLO26n TensorRT 15.60 / 7.53 / 4.57 / 3.80 ms (PyTorch FP32 / TRT FP32 / FP16 / INT8) | Orin Nano Super, 640, batch 1, JetPack 6.1, **MAX power**, **excludes pre/post-processing**, COCO | ⚠️ optimistic bound; use ratios not absolutes |
| YOLO26n TensorRT 13.90 / 7.01 / 4.13 / 3.49 ms (same four formats) | **Orin NX 16 GB**, same conditions as the row above | ⚠️ same caveats; the 4.13 / 4.57 pair is the **1.11×** in §4.1 |
| INT8 costs mAP50-95 0.480 → 0.449; literature 3–7 points, smaller models worse | COCO, not our split, not our size regime | ⚠️ directional only |
| OpenVINO "up to 3× on CPU" | vendor claim, unspecified model/host | ⚠️ vendor |
| Orin Nano Super $249, 67 TOPS, 102 GB/s | vendor spec | ✅ spec, not performance |
| Orin NX 16 GB ~$600–900, 100 TOPS, 102.4 GB/s, 1024 CUDA / 8× A78AE | vendor spec | ✅ spec, not performance |
| TRT 7.2 engines do not load on JetPack 6 / TRT 10 | NVIDIA forums, ProventusNova | ✅ well established |

### ⚠️ Assumed — invented for illustration, must be replaced

| Assumption | Value used | Who supplies the real one |
| --- | --- | --- |
| **Closing speed** | 40 m/s | **the user** |
| **Persistence frames** | 3 | **the user** |
| Pipeline overhead outside detection | 60 ms | measurement, once a rig exists |
| Target airframe size | 0.3 m | the use case |
| Camera HFOV | 60° | the sensor choice |
| Power mode | 15 W | thermal test on the airframe |
| Orin Nano ÷ Xavier NX throughput | 2× (bandwidth) rather than 3.2× (TOPS) | on-device benchmark |
| Orin NX 16 GB ÷ Orin Nano Super, GLAD end to end | ~1.15× overall — 1.11× GPU path, ~1.3–1.5× CPU motion path | on-device benchmark of both |

### Not measured, and worth measuring — in priority order

1. **Per-stage profile of `GladPipeline.step`** on idle hardware — GAD, LAD, MOD2 global,
   MOD2 local, classifier. Cheap (`perf_counter` + ~2,000 contiguous frames), and it is the
   one number this whole document estimates rather than knows.
2. **p50 / p95 / p99 and worst-sequence fps** for EXP-004, re-derived per video from the
   existing run rather than as an aggregate mean.
3. **ONNX/OpenVINO CPU export** — actual speedup on this host, with the re-score.
4. **Everything on the board.** All of §4–§6 is arithmetic until hardware exists.

---

## 9. Where this document sits

- **Requirement (§1)** is blocked on two user decisions and is otherwise illustrative.
- **Decomposition (§2)** is measured and is the document's load-bearing content.
- **Cheap wins (§3)** are actionable now; items 8 and 9 are already banked.
- **Board (§4), alternatives (§5), recommendation (§6)** are a reasoned projection from
  published figures. **No edge hardware has been benchmarked.**
- Update this file whenever a stage is measured, and move the line from §8's assumed table
  into its measured table when it is.
