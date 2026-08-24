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

### The three unknowns

| Input | Status | Who decides |
| --- | --- | --- |
| **Closing speed** | ⚠️ **UNKNOWN — not measured, not specified** | the user |
| **Persistence frames** before the system acts | ⚠️ **UNKNOWN — not measured, not specified** | the user |
| **Is the sensor cued or searching?** | ⚠️ **UNKNOWN** — added when the camera question landed; see [§4.2.8](#428-the-lens-is-a-competing-answer-and-it-is-cheaper) | the user |

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

**Target: a 10-inch-class quadrotor — 0.59 m frontal extent, 0.72 m tip-to-tip diagonal
(supplied by the user 2026-08-23; 0.6 m used below).** 1920 px wide sensor, ~60° horizontal
FOV ⇒ 0.031°/px.

| Apparent size | Range | Time to contact at 40 m/s |
| --- | --- | --- |
| 10 px (first detectable) | **~110 m** | **2.8 s** |
| 30 px | ~37 m | 0.9 s |

⚠️ **Do not carry 0.6 m into a dataset calculation.** ARD-MAV and ARD100 target
Phantom/Mavic-class airframes, ~0.4–0.5 m — smaller than ours. Sanity check on *their*
geometry at 0.45 m: ARD-MAV's p95 closest approach of **34.2 px**
([experiments.md](experiments.md)) is ~24 m, and its smallest band, <8 px, is ~103 m. The
dataset's range span and this arithmetic agree. **The two sizes differ by ~1.3×, and that
factor is the whole of [§4.3.3](#433-projected-through-exp-005s-own-measured-curve)'s
"credit".**

**The whole engagement is on the order of one second.** That is what makes the latency
budget tight, and it is why **input resolution cannot be traded for frame rate**: at 110 m
the target is 10 px in a 1920-wide frame. Halving the input to 960 makes it 5 px and the
detection range collapses. This project's founding trap
([CLAUDE.md](../CLAUDE.md)) wearing a deployment badge.

**Resolution that must be fed: 1920×1080, full rate, contiguous.** Contiguous is not
stylistic — GLAD's motion branches difference against the previous frame, so `src.glad_detect`
has no `--stride` by deliberate design. **Frame striding is not available as a speed lever
on this pipeline.**

> **This table assumes a 1920-px sensor. [§4.2.7](#427-the-upside-stated-fairly--this-is-a-real-win)
> redoes it for a 12 MP Alvium 1800 (4024 px, 0.014911°/px), where first-detectable range
> moves from 55 m to 115 m and the engagement window from 1.4 s to 2.9 s — and
> [§4.2.4](#424-what-12-mp-does-to-glad--the-crux) prices what that costs in frame rate.
> The 60 ms pipeline overhead assumed above is also **too low for a 12 MP sensor**: capture
> alone is ~30–35 ms there, so read it as ~80–90 ms in any 12 MP row.**

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
| 7 | **Frame striding** | **Not available.** | Incoherent for GLAD: both motion branches difference consecutive frames. `src.glad_detect` has no `--stride` on purpose. A strided sample measures a different algorithm. **See item 11 — the objection is to the *gap*, not to processing fewer frames.** |
| 8 | **Tracking between detections** | **Already banked.** | This *is* the local regime. 88.4% of frames already run a crop-sized pass instead of a full-frame one. |
| 9 | **Cache the classifier** | **Already banked.** | Upstream reloaded `Net_best.pth` from disk per candidate per frame, inside a loop running up to 50× per frame. Our port hoists it (`src/algo/glad/classifier.py`, `load_gate`). [glad-model.md §6](glad-model.md) item 1 attributes much of the GMD/LMD cost gap to this. |
| 10 | **Run LAD at native 320 instead of upscaling to 640** | **~4× FLOPs on the modal frame** — the largest single win identified | ⚠️ **This is a retrain, not a config change.** `yolov5s_GLAD-crop.pt` was trained on 320×320 crops fed at 640. Changing the input scale changes the detector. **`algo-agent`'s call**, candidate for M7. |
| 11 | **Duty-cycled inference** (`--sample nth` / `--sample burst`) | **1.55–2.71× at half rate; 10.5–20× as burst pairs.** Measured, EXP-006 to EXP-009 | ✅ **Available, and it is not item 7.** Striding is rejected for the *gap* it opens between differenced frames, not for processing fewer of them. Half rate keeps every processed frame adjacent to its predecessor; a burst differences *inside* a 33 ms pair and sleeps between pairs. Costs are measured against a full-rate control over identical frames — see below. |

**Realistic stack on the target board: (5) × (3) ≈ 3.4× over on-board PyTorch FP32**,
before any architectural change, at a re-scoring cost of two `src.evaluate` runs over a
persisted JSONL.

### Item 11 in detail — what a duty cycle actually costs

Measured 2026-08-24, EXP-006 to EXP-009. Each policy is scored against a **full-rate
control over the identical frames** (`src.evaluate --keys-from`), so the duty cycle is
the only variable — same weights, same `--pad released`, same criterion.

| Policy | ARD-MAV retained | ARD100 retained | Compute saved |
| --- | --- | --- | --- |
| **Half rate** (`--sample nth --sample-n 2`) | **91.5%** | **91.5%** | 2.71× / 1.55× |
| **Burst pairs** (`--sample burst`, K=2) | 48.2% | **36.0%** | 20× / 10.5× |

Three findings that change how this lever should be used:

- **Half rate's cost is a property of the policy, not the content.** 91.5% on both
  splits, to three significant figures. It can be quoted as one number and does **not**
  compound with the generalisation gap. Burst pairs does compound — 48.2% where GLAD is
  strong, 36.0% on video it never trained on — because cold acquisition leans on GAD,
  which is exactly what fails to generalise (EXP-009: motion carries 77% of burst
  detections on ARD100 against 50% on ARD-MAV).
- **The compute saving is content-dependent and is not the frame saving.** Half the
  frames bought 2.71× on ARD-MAV but only 1.55× on ARD100, because losing lock more often
  means paying the motion path more often. **Never quote a duty cycle's saving as its
  duty-cycle ratio.**
- **Burst pairs is disqualified by latency, not by recall.** At 25.0% per burst the
  expected wait for a first detection is four bursts. At the 30-second period the scheme
  was proposed with, that is **120 s — 4,806 m of closing at 40 m/s.** A scheme costing
  0.6% of the compute is still unusable if the target crosses the whole engagement
  envelope before it is seen.

**Where this leaves the lever.** Half rate is a genuine, cheap, measured win and is worth
taking on the target board. Burst pairs is a *fallback for hardware that cannot sustain
half rate*, never a power-saving choice — which is precisely how `src.live_detect
--policy auto` applies it ([live_detect.md](live_detect.md)).

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

### 4.2 The camera — Allied Vision Alvium 1800 at 12 MP

**The document had no camera model at all until now, and that was a real gap: capture is
the first term in the end-to-end chain [§1](#1-the-requirement) defines, and at 12 MP it
stops being a rounding error.**

#### 4.2.1 Which camera, exactly — the answer differs by variant

"12 MP Allied Vision 1800" resolves to **four** Alvium 1800 products, and they are not
interchangeable. Vendor datasheet figures, at full resolution:

| Variant | Sensor | Resolution | MP | Shutter | Interface | **Max fps @ full res** | Sensor depth | Power |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **1800 U-1240** | Sony IMX226 | 4024 × 3036 | 12.22 | Rolling (+ global *reset*) | USB3 Vision | **29** | 8 / 10-bit | ~2.5 W |
| **1800 C-1240** | Sony IMX226 | 4024 × 3032 | 12.20 | Rolling (+ global *reset*) | **MIPI CSI-2, 4 lanes** | **41** | 10-bit | 2.9 W |
| **1800 U-1236** | Sony IMX304 | 4112 × 3008 | 12.37 | **Global** | USB3 Vision | **22–23** | 12-bit | — |
| **1800 C-1236** | Sony IMX304 | 4112 × 3008 | 12.37 | **Global** | **MIPI CSI-2, 4 lanes** | **22** | 12-bit | 2.6 W |

> **Budgeted below: the `1800 C-1240` (IMX226, CSI-2, 41 fps).** It is the fastest of the
> four and therefore the honest worst case for compute. **[§4.2.5](#425-rolling-vs-global-shutter--the-point-that-outranks-frame-rate)
> argues the C-1236 is probably the better buy anyway, for a reason that is not about
> speed.**

#### 4.2.2 The sensor ceiling, and where the link actually caps

**CSI-2 (C-1240) — the sensor is the limit, not the link.**

```
4024 x 3032              = 12,200,768 px  x 10 bit = 122.0 Mbit/frame
122.0 Mbit x 41 fps      = 5.00 Gbit/s
Orin NX 4-lane D-PHY 2.1 = 4 x 2.5        = 10.0 Gbit/s
```

**50% link utilisation at the sensor's maximum** — two-times headroom. The 41 fps is a
sensor readout limit; CSI-2 is nowhere near binding.

**USB3 (U-1240) — the link caps first, and the bit depth chooses the frame rate.**
USB3 Vision sustains roughly **350–400 MB/s** of payload in practice, not the 5 Gbit/s the
PHY advertises:

| Pixel format | Bytes/frame | fps at 350 MB/s | Published |
| --- | --- | --- | --- |
| Mono8 / Bayer8 | 12.22 MB | **28.6** | **29** ✅ |
| RAW10 (packed) | 15.27 MB | **22.9** | — |

**The published 29 fps is the USB3 bandwidth cap at 8 bit, not a sensor figure.** Asking
for 10-bit over USB3 costs ~21% of the frame rate. This is precisely why a frame-rate
number without its pixel format and bit depth is incomplete.

**IMX304 (the 1236 pair) is 22 fps on *both* interfaces**, so there the sensor binds and
the interface choice carries no frame-rate consequence.

#### 4.2.3 Can the Orin NX ingest it? — yes, and this path is *supported*

| Question | Answer | Status |
| --- | --- | --- |
| CSI-2 lanes on Orin NX | Two 4-lane **or** four 2-lane D-PHY groups; 2.5 Gbit/s per lane, **20 Gbit/s aggregate** | ✅ vendor datasheet |
| Lanes the Alvium driver needs | **4 lanes only — no 2-lane fallback** | ⚠️ carrier must expose a 4-lane connector |
| ISP throughput | **1.75 GPixel/s**; raw Bayer sensors up to **24 MP** | ✅ |
| ISP load at 12.2 MP × 41 fps | 500 MPixel/s = **29% of capacity** | ✅ comfortable |
| Driver | `alliedvision/alvium-jetson-driver-release` — **JetPack 6.2**, `.deb` via APT, **all Orin modules incl. Orin NX**, V4L2 + Vimba X | ✅ |

> **Worth noting against the rest of this document:** the ingest side is the *one* part of
> this deployment that is a vendor-supported, currently-maintained path. That is the
> opposite of [§4's toolchain break](#-toolchain-break--load-bearing-and-it-is-not-an-export),
> where GLAD's TensorRT 7.2 entry point is dead and must be rebuilt. **The camera is not
> the risk here. The detector runtime is.**

**Does USB3 ingest cost CPU that GLAD needs? Yes — and it lands on the wrong branch.**
USB3 Vision payload assembly and memcpy happen in the host driver and SDK, typically
**15–30% of a core** at 350 MB/s plus interrupt load. GLAD's expensive branch — KLT,
pyramidal Lucas–Kanade, RANSAC homography, warp, morphology — is **CPU-bound OpenCV**. USB3
therefore taxes exactly the resource the bottleneck runs on, while CSI-2 DMAs into memory
at near-zero CPU. **[ASSUMED — vendor-typical figures, not measured on this rig.]**

**Capture latency is now a first-order term.** At 41 fps the rolling readout occupies
~24 ms of every frame period. Add CSI DMA, debayer and grayscale conversion (~8 ms at
12.2 MP) and **capture alone is ~30–35 ms before the detector starts** — comparable to the
entire inference. §1's illustration **[ASSUMES] 60 ms** of non-detection pipeline; at 12 MP
that is optimistic and should be read as **~80–90 ms**.

#### 4.2.4 What 12 MP does to GLAD — the crux

**Every number in [experiments.md](experiments.md), and every projection elsewhere in this
document, assumes 1920×1080.** 4024×3032 is **5.88× the pixels**. Decomposed against the
branch distribution from [§2.3](#23-inside-glad--why-the-state-machine-is-not-the-problem):

| Branch | Share | Scales with sensor resolution? | Why |
| --- | --- | --- | --- |
| **`local yolo` (LAD only)** | **88.4%** | **Mostly not** | The search region is a fixed **320×320 window in sensor pixels**, upscaled to 640. Its inference cost is `O(crop)` — independent of frame size. |
| — but its per-frame overhead | | **Yes, 5.88×** | Every frame must still be captured, debayered, converted to grayscale and **stored as the previous frame** for possible differencing. That is full-frame work on every frame, whether or not the motion branch fires. |
| **Motion path** (`local mod`, `local miss`, `global miss`, `global mod`) | **11.6%** | **Yes, ~5.88× — and this was already the bottleneck** | KLT grid, pyramidal LK, RANSAC homography, full-frame warp, absdiff, median blur, 3× morphological open/close. All `O(pixels)`. |
| **`global yolo` (GAD)** | 0.1% | Constant cost, **worse result** | A 640 letterbox of a 4024-px frame is a **6.3× reduction**, against 3× at 1080p. |

> **The founding trap, doubled.** [CLAUDE.md](../CLAUDE.md)'s rule is that a 640 letterbox
> of a 4K frame shrinks a 20 px drone to ~3 px. At 1080p GAD already reduces 3×; at 12 MP
> it reduces **6.3×**, so a target arriving at 21 px reaches the global detector at
> **3.3 px** — below the stride-8 P3 cell, which is the structural reason GAD's published
> recall is 0.17 ([glad-model.md §2](glad-model.md)). **GAD gets strictly worse at 12 MP,
> and GAD is the acquisition branch** — the one EXP-004 already identified as the real
> ceiling. That is not a cost you pay; it is a capability you lose.

**And the pixel constants break.** [glad-model.md §6](glad-model.md) item 4 already records
that `area 30–3000`, `a = 160`, `dist_ref = 200`, blur kernel 11 and `local_num == 30` are
**absolute pixels tuned for 1920×1080**, and that anything at another resolution *silently
mis-filters*. It is flagged there as a threat to M4b. **At 12 MP every one of them is wrong
by 2.1× linear / 5.9× in area, and normalising by frame diagonal stops being advisable and
becomes mandatory.**

##### Projected frame rate **[EXTRAPOLATED — no board and no camera has been benchmarked]**

Model: take §4.1's Orin NX 16 GB rows, hold the LAD inference constant (crop-sized, hence
resolution-independent), and scale the full-frame terms by 5.88×. Same conditions as §4.1 —
**TensorRT FP16, batch 1, MAX power, pre/post-processing excluded from the vendor anchor.**

| Row | Orin NX @ 1080p (§4.1) | **@ 12 MP, naive full-frame** | **@ 12 MP, hybrid (§4.2.6)** |
| --- | --- | --- | --- |
| **Modal frame** (88.4%, LAD only) | ~65 fps | **~29 fps** | ~34 fps |
| **Motion path firing** (11.6%) | **14–19 fps** | ⚠️ **2.4–3.2 fps** | ~7–9 fps |
| **Sustained mean over content** | ~50–55 fps | ⚠️ **~13–14 fps** | **~25 fps** |

> ### ⚠️ The hard-scene row fails.
>
> **2.4–3.2 fps in the scenes where the motion path fires.** Against §1's illustration
> (40 m/s closing, N = 3 persistence, and now ~85 ms of capture-and-pipeline overhead),
> hard-scene detection latency becomes `3/3 + 0.085 ≈ 1.09 s`, during which the target
> closes **~44 m**. §4.2.7 puts the whole engagement at 12 MP at ~2.9 s, so **the hard
> scene spends 38% of the entire engagement in latency alone** — at exactly the moment
> detection mattered.
>
> Naive full-resolution 12 MP into GLAD is **not viable**. Not because 13 fps mean is slow,
> but because the mean is not the number: the pipeline degrades ~4.5× precisely when the
> scene is hard, and 5.88× more pixels multiplies the degradation rather than the average.

#### 4.2.5 Rolling vs global shutter — the point that outranks frame rate

**This probably matters more than any figure above, and it is not a speed argument.**

The IMX226 variants (**U/C-1240**) are **rolling shutter**; the IMX304 variants
(**U/C-1236**) are **global shutter**. The camera is going on a *flying* drone.

**Rolling shutter breaks the motion module's core assumption.** GLAD compensates ego-motion
by fitting a **RANSAC homography** between consecutive frames and warping the previous frame
onto the current one. A homography is a *global* 2D projective transform — it assumes every
pixel in the frame was captured at the same instant. Under rolling shutter with fast camera
rotation, rows are captured at different times and the true warp is **not** a homography.
**The residual after warping is exactly what GLAD thresholds as "motion"**, so rolling
shutter injects structured false motion directly into the branch that produces the
detections — most strongly during rapid attitude change, which is precisely when an
interceptor manoeuvres.

The "Global Reset Shutter" mode the 1240 advertises does **not** rescue this: it starts all
rows together but ends them sequentially, so it needs strobed illumination and is useless
in daylight.

**The honest counter-argument, from our own data.** ARD-MAV and ARD100 were shot on DJI
Mavic 2 / M300 gimbal cameras, which are **rolling shutter** — so EXP-004's 0.81 F1 was
achieved on rolling-shutter footage from a moving platform. Rolling shutter is demonstrably
workable here. **But a gimbal-stabilised Mavic is a far gentler platform than a hard-mounted
camera on an interceptor**, and nothing in this project measures the hard-mounted case.
Read it as: *proven on a gimbal, unproven hard-mounted.*

| | C-1240 (rolling) | C-1236 (global) |
| --- | --- | --- |
| Max fps @ 12 MP | **41** | 22 |
| Homography assumption | ⚠️ violated under rotation | ✅ holds |
| Sensor format | 1/1.7" | **1.1"** — far larger pixels, better low light |
| Relevance to M2b's finding | — | ✅ M2b measured **19 points of small-target recall lost** at <5 grey levels of target/background separation; a 1.1" sensor attacks that directly |

**If the airframe is hard-mounted or manoeuvres hard, take the C-1236 and its 22 fps.** The
pipeline's sustained projection is ~13–25 fps anyway
([§4.2.4](#projected-frame-rate-extrapolated--no-board-and-no-camera-has-been-benchmarked)),
so the sensor's 41 fps was never going to be reached — **the C-1240 is buying frame rate the
compute cannot consume, at the cost of an assumption the algorithm depends on.**

#### 4.2.6 What to actually do with the pixels

Three ways to spend a 12 MP sensor, and only one of them is good:

**(a) Feed GLAD the full 12 MP.** ❌ Fails — see §4.2.4. It also degrades GAD, the
acquisition branch, by widening its letterbox reduction to 6.3×.

**(b) 2×2 bin to 2012×1516.** ⚠️ Throws away the entire resolution benefit — binned angular
resolution is 0.0298°/px against 1080p's 0.0313°/px, i.e. **the camera you already
assumed**. Not pointless, but the benefit is a different one from the one being bought:
binning sums four photosites for ~4× signal and ~2× SNR, which attacks **M2b's measured
19-point small-target recall loss under low target/background contrast**. Buy binning for
SNR, never for resolution.

**(c) Full readout; native-resolution crop for LAD, downsampled copy for the motion module.**
✅ **This is the recommendation.** It exploits the structure §2.3 already established:

- **88.4% of frames** run LAD on a **320×320 native-resolution crop** — the target arrives
  at full 12 MP angular resolution, **2.1× more pixels on target than 1080p can ever give**,
  at zero extra inference cost, because the crop is a fixed pixel size.
- **11.6% of frames** run the motion module on a **2×2 binned / downsampled copy**, so the
  `O(pixels)` branch never pays the 5.88×.
- **GAD** runs on that same downsampled copy, restoring its letterbox reduction to the 3× it
  was tuned at instead of 6.3×.

Projected **~25 fps sustained, ~7–9 fps hard scene** — the third column of §4.2.4's table.
Still degraded, still to be checked against a real requirement, but no longer failing.

> ### ⚠️ Option (c) is a protocol change, not a configuration.
>
> `yolov5s_GLAD-crop.pt` was trained on **320×320 crops taken from 1080p frames**. A native
> 320×320 crop of a 12 MP frame holds a target ~2.1× larger in pixels. **That is an
> input-scale change, and by this project's own rule ([CLAUDE.md](../CLAUDE.md), and §7
> here) it means the detector is no longer the detector that was measured.** Expect it to
> need a **retrain**, not merely a re-score — and even the re-score verdict is
> `algo-agent`'s, not this document's. The same applies to every absolute-pixel constant in
> `MOD2.py`.
>
> **No accuracy number in this repository may be read as applying to a 12 MP build.**

**A fourth option exists and should not be dismissed: sensor-side ROI readout.** Alvium
supports windowed readout, and a smaller ROI reads out *faster*. Running the local regime as
a sensor-side 320×320 ROI would cut transfer and capture latency to almost nothing. It is
blocked on one unmeasured quantity — **ROI mode-switch latency**, typically several frames,
paid on every regime transition. EXP-004 is encouraging here: acquisition happened only
**42 times in 28,337 frames**, so the regime is stable in practice. Harder content could
thrash it. Filed as a todo.

#### 4.2.7 The upside, stated fairly — this is a real win

12 MP is not only cost. At the **same FOV**, 4024 px against 1920 px is **2.096× finer
linearly**, and that lands squarely on the project's hardest failure mode.

**0.6 m target** (10-inch class, §1), ~60° horizontal FOV. 1080p: 0.03125°/px. C-1240:
**0.014911°/px**. The right-hand column is in *apparent pixels* and so is unaffected by
target size.

| Apparent size | Range @ 1080p | **Range @ 12 MP** | What EXP-004 measures at that size |
| --- | --- | --- | --- |
| 10 px | 110.0 m | **230.6 m** | `tiny` — recall 0.849 (centre@1×) |
| 16 px | 68.8 m | **144.1 m** | boundary of the 0.984 band |
| 20 px | 55.0 m | **115.3 m** | `small` — recall **0.984** |
| 30 px | 36.7 m | **76.9 m** | `small` — recall 0.984 |

> **The headline: 12 MP moves the 0.98-recall boundary from ~69 m out to ~144 m**, and
> first-detectable from 110 m to 231 m. A target at 110 m is a marginal 10 px at 1080p and a
> comfortable 21 px at 12 MP. **18,265 of ARD-MAV's 28,160 targets are under 16 px** — this
> sensor moves a large part of that population out of the band where the detector is
> weakest, which is a more direct attack on the project's central problem than any
> architecture change currently on the table.

Time-to-contact at the **[ASSUMED]** 40 m/s closing speed: **2.8 s at 1080p, 5.8 s at
12 MP.** The sensor roughly **doubles the reaction time available**, which is what partly
pays for the latency it costs.

**The trade, honestly:**

| | 1080p | 12 MP naive | **12 MP hybrid (c)** |
| --- | --- | --- | --- |
| Sustained fps (Orin NX) | ~50–55 | ~13–14 | ~25 |
| Hard-scene fps | 14–19 | ⚠️ 2.4–3.2 | ~7–9 |
| First-detectable range | 55 m | **115 m** | **115 m** |
| Engagement window | 1.4 s | **2.9 s** | **2.9 s** |
| Hard-scene latency ÷ window | 19% | ⚠️ **38%** | **~15%** ✅ |

**Option (c) is the only column that improves both the range and the latency ratio.**

**Whether the trade is worth taking is `algo-agent`'s call, not this document's** — it hinges
on accuracy at a changed input scale, which is unmeasured. The pixel arithmetic above is the
part that belongs here.

#### 4.2.8 The lens is a competing answer, and it is cheaper

The same angular resolution as 12 MP @ 60° is obtainable from **1080p with a 28.6° lens**
(60 ÷ 2.096): identical °/px, **5.88× less compute**, no protocol change, no new sensor.
What it costs is **field of view — 2.1× narrower, so 4.4× less solid angle searched.**

**Which is right depends on a question nobody has answered:**

> ### ⚠️ Third unknown, alongside closing speed and persistence frames
>
> **Is the sensor cued or searching?**
>
> - **Cued** (radar / RF / GCS handoff supplies a bearing) → **a narrow lens on 1080p is
>   strictly better**: same pixels on target, a fraction of the compute, no retrain, no new
>   camera. 12 MP would be buying FOV that is not being used.
> - **Searching blind** → the FOV *is* the requirement, and 12 MP is the way to have both
>   FOV and pixels on target. The compute cost is then unavoidable, and option (c) is how to
>   pay it.
>
> **This is the user's call.** It changes the recommendation completely and it is cheap to
> answer.

#### 4.2.9 Camera recommendation

1. **Answer the cued-vs-searching question first** (§4.2.8). If cued, buy a narrower lens
   and keep 1080p — cheaper on every axis, and no retrain.
2. **If searching: take a CSI-2 variant, not USB3.** 41 vs 29 fps, DMA instead of a 15–30%
   CPU tax on the branch that is CPU-bound, lower capture latency, less cabling on an
   airframe. Requires a carrier exposing a **4-lane** CSI connector — the driver has no
   2-lane mode.
3. **Prefer the global-shutter C-1236 over the C-1240** unless the camera is gimballed. Its
   22 fps sits above the ~13–25 fps the pipeline can sustain anyway, so the C-1240's extra
   frame rate is unusable, while its rolling shutter violates the homography assumption
   GLAD's motion compensation is built on. The 1.1" sensor is a second, independent win
   against M2b's contrast finding.
4. **Never feed GLAD the full 12 MP.** Use option (c): native crop for LAD, downsampled copy
   for the motion module and GAD.
5. **Budget a retrain, not a re-score.** Option (c) changes LAD's input scale by 2.1× and
   invalidates every absolute-pixel constant in `MOD2.py`. That is M7 work, and the accuracy
   verdict is `algo-agent`'s.
6. **Measure capture latency before anything else.** It is ~30–35 ms at 12 MP — comparable
   to the whole inference — and §1's 60 ms pipeline assumption understates it.

**What would change this:** a measured ROI mode-switch latency under ~2 frames would make
sensor-side windowing beat option (c); a gimballed mount removes the global-shutter argument
and makes the C-1240 the pick; and a cued sensor removes the case for 12 MP altogether.

---

### 4.3 The opposite direction — an analog FPV camera (iFlight RaceCam R1 Mini)

**Asked 2026-08-23: "what would my model results be, using this camera?" The answer is
near-zero recall, and the arithmetic is not close.** This section exists so the question is
not re-asked, and because it is the cleanest available demonstration of why §1's resolution
floor is not negotiable.

#### 4.3.1 What the camera is

Two variants ship under the name; both are **analog CVBS**, not a digital sensor interface.

| | R1 Mini (CCD) | R1 Mini 1200TVL |
| --- | --- | --- |
| Sensor | 1/3" CCD, 600 TVL | 1/1.8" starlight, 1200 TVL |
| Lens / horizontal FOV | 2.5 mm / **130°** | 2.1 mm / **165°** |
| Output | **CVBS**, PAL/NTSC, 4:3 | **CVBS**, PAL/NTSC, 4:3 & 16:9 |
| Frame delivered after capture | 720×576i @ 25 / 720×480i @ 30 | same |
| Shutter | electronic rolling | electronic rolling |

**TVL is not pixels.** It is a resolvable-line count on the sensor side; what reaches a
capture card is a composite analog line, digitised to **720 active samples**, interlaced.
The 1200 TVL number cannot survive a 7–8 MHz composite link — 720 px is the generous
reading, and every row below uses it.

#### 4.3.2 Angular resolution — the whole answer

ARD100 was shot on **DJI Mavic 2 / M300** gimbal cameras at 1920×1080 (≈28 mm equivalent,
**65.5° horizontal**) ⇒ **29.3 px/°**. Against that:

| Camera | px/° | Coarser by | EXP-005 median target | p95 |
| --- | --- | --- | --- | --- |
| **ARD100 as shot** (1920 px / 65.5°) | 29.31 | 1.00× | **14.7 px** | 27.2 px |
| R1 Mini CCD (720 px / 130°) | 5.54 | **5.29×** | **2.8 px** | 5.1 px |
| R1 Mini 1200TVL (720 px / 165°) | 4.36 | **6.72×** | **2.2 px** | 4.1 px |
| R1 Mini 1200TVL, 960-px digitiser (generous) | 5.82 | 5.04× | 2.9 px | 5.4 px |

**Every target in EXP-005's corpus lands between 1 and 5 px.** The whole distribution moves
below the smallest size bin in which GLAD scores a single true positive.

#### 4.3.3 Projected through EXP-005's own measured curve

Fine-binned recall from `runs/exp005_glad_ard100/matches_center.csv` — 33,517 targets, the
same dump the [EXP-005](experiments.md) metric block is built from:

| gt size (px) | n | recall |
| --- | ---: | ---: |
| 4–6 | 4 | **0.0000** |
| 6–8 | 300 | **0.1600** |
| 8–10 | 1,850 | 0.3481 |
| 10–12 | 5,270 | 0.5268 |
| 12–14 | 6,975 | 0.6346 |
| 14–16 | 6,218 | 0.7189 |
| 16–20 | 6,712 | 0.7978 |
| 20–24 | 3,244 | 0.8813 |
| 24–32 | 2,201 | 0.9082 |
| 32–48 | 653 | 0.6524 |
| 48–96 | 77 | 0.7013 |
| >96 | 13 | 0.1538 |

Rescaling every target by the shrink factor and reading off the bin it lands in:

| Camera | targets ≥6 px | ≥10 px | **projected recall** |
| --- | ---: | ---: | ---: |
| ARD100 as shot | 100.0% | 93.6% | 0.688 *(= the measured EXP-005 figure)* |
| R1 Mini CCD, 130° | 2.3% | 0.2% | **≈0.005** |
| R1 Mini 1200TVL, 165° | 0.6% | 0.1% | **≈0.001** |

The middle column is the honest one: **97.7% of targets fall below the size at which GLAD
has ever detected anything.** The projected-recall column is arithmetic on top of that and
should be read as "indistinguishable from zero", not as three significant figures.

**The size "credit", and why it is small.** The table above rescales *measured pixels*, so
it answers "re-shoot ARD100's own engagements through this camera" and is independent of how
big the target physically is. Our target is a **10-inch quad at ~0.59 m** (§1) while ARD100
flies **Phantom/Mavic-class, ~0.4–0.5 m**, so at equal range ours is ~1.2–1.4× more pixels.
Dividing the shrink factor by that credit and re-reading the curve:

| credit | R1 Mini 130°: ≥8 px / recall | R1 Mini 165°: ≥8 px / recall |
| --- | ---: | ---: |
| 1.0× (ARD100's own targets) | 0.5% / 0.005 | 0.2% / 0.001 |
| **1.2×** | 1.3% / **0.013** | 0.4% / **0.004** |
| **1.33×** | 2.2% / **0.020** | 0.6% / **0.006** |
| 2.0× (a 0.9 m target — *not* ours) | 14.1% / 0.103 | 5.2% / 0.042 |

**A 10-inch target moves projected recall from 0.005 to about 0.013–0.020.** It is a real
factor and it changes nothing: 6.7× of angular resolution is not bought back by 1.3× of
airframe. The 2.0× row is included only to show where the curve would start to move — and
0.9 m is a different aircraft, not a bigger quad.

#### 4.3.4 Detection range, which is the number that matters

**0.6 m target** (10-inch class, §1), so apparent size = px/° × 34.38 / R. **8 px is
GLAD's measured floor** — below it recall is under a third, and below 6 px there is not one
true positive in 33,517 targets:

| Camera | 6 px | **8 px (GLAD's floor)** | 10 px | 12 px | 20 px |
| --- | ---: | ---: | ---: | ---: | ---: |
| ARD100 optics | 168.0 m | **126.0 m** | 100.8 m | 84.0 m | 50.4 m |
| R1 Mini CCD, 130° | 31.7 m | **23.8 m** | 19.0 m | 15.9 m | 9.5 m |
| R1 Mini 1200TVL, 165° | 25.0 m | **18.8 m** | 15.0 m | 12.5 m | 7.5 m |

At §1's 40 m/s closing speed, the 8 px floor gives **3,149 ms of warning on ARD100's optics
and 469–595 ms on this camera.** That does clear the 274 ms hard-scene pipeline latency
budgeted for an Orin Nano — **on a 10-inch target the timing argument is tight but not
fatal, and this section does not rest on it.** What is fatal is §4.3.3: reaching 8 px at
19–24 m is worth little when only 0.4–2.2% of an engagement's frames get there, and 8 px is
where GLAD recalls 0.348. It is optics, and no board, quantisation or retrain recovers it.

#### 4.3.5 Four more reasons, all pushing the same way

The size arithmetic alone is disqualifying, so these only matter for not overstating the
projection above — **each makes the real number worse, never better**:

1. **The 320×320 search region stops being local.** `REGION_HALF = 160`
   (`src/algo/glad/pipeline.py:44`) is in *sensor* pixels. On a 720×576 frame that window is
   **44% of the frame width** instead of 17%, so LAD degrades toward being a whole-frame
   640 detector — EXP-001's failure mode, reintroduced by geometry.
2. **Frame differencing against analog noise.** GAD and the motion module difference
   consecutive frames. CVBS brings per-line noise, AGC flicker, chroma crawl and interlace
   comb; a 2 px target has no SNR against that. This attacks the branch EXP-005 already
   shows failing — `global miss` at 12.3%.
3. **Rolling shutter on a hard-mounted racing airframe** — the case [§4.2.5](#425-rolling-vs-global-shutter--the-point-that-outranks-frame-rate)
   flags as unmeasured and worst-case, since the homography behind motion compensation
   assumes a single capture instant.
4. **165° of barrel distortion.** Nothing in the pipeline undistorts, and the homography is
   fitted assuming it does not need to.

#### 4.3.6 If the camera is fixed, the lens is the only lever

Swapping the 2.1 mm for a long lens is the one cheap move — it is
[§4.2.8](#428-the-lens-is-a-competing-answer-and-it-is-cheaper)'s argument again. But
**720 px is a hard ceiling**: even at a 30° FOV the camera gives 24 px/° against 1080p's
29.3, so it still never reaches the optics EXP-005 was measured on, and a 30° search cone
demands the cueing question of [§4.2.8](#428-the-lens-is-a-competing-answer-and-it-is-cheaper)
be answered *yes*.

**Recommendation: do not put this camera in the detection path.** It is an FPV pilot's
video-link camera — low latency, low light, wide angle, human in the loop — and every one of
those choices is the opposite of what small-target detection needs. Keep it as the pilot
feed if the airframe wants one, and run detection off a separate digital sensor
([§4.2.9](#429-camera-recommendation)). **The minimum for GLAD as measured is ~1920 px
across ≲70°, digital, global shutter.**

> **[PROJECTED — no footage from this camera has been run.]** §4.3.3 rescales EXP-005's
> measured size/recall curve and assumes apparent size is the only variable. It is not;
> §4.3.5 lists four effects that all push downward, so **treat these numbers as an upper
> bound.** To measure it instead: capture CVBS to file, label it, and run
> `src.glad_detect` + `src.evaluate` exactly as EXP-005 did.

---

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
| **Sensor is cued rather than searching** | **Drop 12 MP.** A 28.6° lens on 1080p gives identical pixels-on-target for 5.88× less compute and no retrain ([§4.2.8](#428-the-lens-is-a-competing-answer-and-it-is-cheaper)) |
| **Camera is hard-mounted / manoeuvres hard** | **Global-shutter C-1236 over C-1240**, accepting 22 fps — rolling shutter violates the homography assumption GLAD's ego-motion compensation depends on ([§4.2.5](#425-rolling-vs-global-shutter--the-point-that-outranks-frame-rate)) |
| **12 MP is fixed by procurement** | Never feed it whole. Option (c) — native crop for LAD, downsampled copy for motion and GAD — and **budget a retrain, not a re-score** ([§4.2.6](#426-what-to-actually-do-with-the-pixels)) |

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
| Fine-binned recall vs target size, 4→96+ px, 33,517 targets | EXP-005 dump; §4.3.3 |
| Target airframe: 10-inch class, 0.59 m frontal / 0.72 m diagonal | **supplied by the user, 2026-08-23** |
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
| **Alvium 1800 U-1240: 4024×3036, IMX226 rolling, USB3, 29 fps, 8/10-bit** | vendor datasheet, full res; the 29 fps is the **USB3 payload cap at 8-bit**, ~23 fps at RAW10 | ✅ spec |
| **Alvium 1800 C-1240: 4024×3032, IMX226 rolling, CSI-2 4-lane, 41 fps, 10-bit, 2.9 W** | vendor datasheet, full res; 5.00 Gbit/s = **50% of a 4-lane link** | ✅ spec |
| **Alvium 1800 U-1236 / C-1236: 4112×3008, IMX304 global shutter, 22–23 fps, 12-bit, 2.6 W** | vendor datasheet; sensor-limited, so identical on both interfaces | ✅ spec |
| Orin NX CSI-2: two 4-lane or four 2-lane D-PHY, 2.5 Gbit/s per lane, 20 Gbit/s aggregate | NVIDIA datasheet | ✅ spec |
| Orin NX ISP 1.75 GPixel/s; raw Bayer sensors to 24 MP | NVIDIA datasheet | ✅ spec |
| Alvium Jetson driver: JetPack 6.2, `.deb`, all Orin modules, V4L2 + Vimba X, **4 lanes only** | `alliedvision/alvium-jetson-driver-release` | ✅ supported path |

### ⚠️ Assumed — invented for illustration, must be replaced

| Assumption | Value used | Who supplies the real one |
| --- | --- | --- |
| **Closing speed** | 40 m/s | **the user** |
| **Persistence frames** | 3 | **the user** |
| Pipeline overhead outside detection | 60 ms | measurement, once a rig exists |
| Camera HFOV | 60° | the sensor choice |
| Power mode | 15 W | thermal test on the airframe |
| Orin Nano ÷ Xavier NX throughput | 2× (bandwidth) rather than 3.2× (TOPS) | on-device benchmark |
| Orin NX 16 GB ÷ Orin Nano Super, GLAD end to end | ~1.15× overall — 1.11× GPU path, ~1.3–1.5× CPU motion path | on-device benchmark of both |
| **Which Alvium variant** | **C-1240 budgeted** (fastest ⇒ worst case for compute); §4.2.5 argues for C-1236 | **the user** |
| **Is the sensor cued or searching?** | ⚠️ **UNKNOWN — third unresolved input.** Decides 12 MP vs a narrow lens on 1080p outright (§4.2.8) | **the user** |
| Camera mount | hard-mounted assumed; a gimbal removes the rolling-shutter objection | the airframe design |
| USB3 Vision sustained payload | 350–400 MB/s | measurement on the board |
| USB3 ingest CPU cost | 15–30% of a core | measurement on the board |
| Capture + debayer + grayscale at 12.2 MP | ~30–35 ms | measurement on the board |
| Full-frame terms scale linearly with pixel count | 5.88× (12.20 MP ÷ 2.07 MP) | the per-stage profile, item 1 below |
| 12 MP GLAD projections (29 / 2.4–3.2 / 13–14 fps naive; 34 / 7–9 / 25 hybrid) | arithmetic on §4.1, itself arithmetic on a vendor benchmark | **[EXTRAPOLATED] — board + camera in hand** |
| ROI mode-switch latency on Alvium | unknown; assumed "several frames" | vendor test |

### Not measured, and worth measuring — in priority order

1. **Per-stage profile of `GladPipeline.step`** on idle hardware — GAD, LAD, MOD2 global,
   MOD2 local, classifier. Cheap (`perf_counter` + ~2,000 contiguous frames), and it is the
   one number this whole document estimates rather than knows.
2. **p50 / p95 / p99 and worst-sequence fps** for EXP-004, re-derived per video from the
   existing run rather than as an aggregate mean.
3. **ONNX/OpenVINO CPU export** — actual speedup on this host, with the re-score.
4. **The 5.88× pixel-scaling assumption**, which the whole of §4.2.4 rests on. Item 1's
   profile answers it for free if it is run at two resolutions instead of one — downscale
   ARD-MAV to 960×540 and confirm the motion path falls ~4×. **No camera required.**
5. **ROI mode-switch latency** on the Alvium — decides whether sensor-side windowing beats
   §4.2.6's option (c).
6. **Everything on the board.** All of §4–§6 is arithmetic until hardware exists.
7. **Any real footage from the intended airframe's camera.** §4.3 projects an analog FPV camera to near-zero recall from optics alone, but the projection has never been checked against a frame this project actually captured — every accuracy number here comes from someone else's gimbal.

---

## 9. Where this document sits

- **Requirement (§1)** is blocked on two user decisions and is otherwise illustrative.
- **Decomposition (§2)** is measured and is the document's load-bearing content.
- **Cheap wins (§3)** are actionable now; items 8 and 9 are already banked.
- **Board (§4), alternatives (§5), recommendation (§6)** are a reasoned projection from
  published figures. **No edge hardware has been benchmarked.**
- **Camera (§4.2)** is vendor datasheet specification plus arithmetic on §4.1. The sensor,
  link, ISP and driver facts are solid; **every frame-rate projection through GLAD at 12 MP
  is [EXTRAPOLATED]** and rests on the untested assumption that full-frame stages scale
  linearly with pixel count. **No camera has been benchmarked either.**
- Update this file whenever a stage is measured, and move the line from §8's assumed table
  into its measured table when it is.
