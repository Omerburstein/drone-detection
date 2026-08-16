# Dataset reference

Every drone dataset considered for this project: what is in it, how it was physically
captured, what the labels say, how to get it, and whether it fits.

[research-notes.md](research-notes.md) is the shortlist and the plan. **This file is the
detail.** Facts I could not confirm from a primary source are marked *(unverified)* rather
than guessed.

---

## The three hard filters

The end system is a camera **on a multirotor**, detecting **other multirotors**. That gives
three independent requirements, and most "drone datasets" fail at least one:

| # | Filter | Why it is load-bearing |
| --- | --- | --- |
| 1 | **Target is a multirotor** | A quadcopter silhouette is roughly square (aspect ~1:1–1.5:1) and its motion signature is hover-capable. A fixed-wing target is a 3:1 wing planform that cannot hover and shows a distinctive V-shape blur. Detectors trained on one do not transfer to the other. |
| 2 | **Video, not stills** | Every method that works at this target size (GLAD, YOLOMG, Dogfight, TransVisDrone) uses inter-frame motion. Sparse stills cannot drive a motion branch at all. |
| 3 | **Camera on a moving airborne platform** | A static camera makes frame differencing trivial and removes the ego-motion that motion compensation exists to cancel. Ground footage also lacks the sky/terrain background swing that causes our false positives. |

**Only four real datasets pass all three.** They are in Tier 1 below.

---

## Fit summary

| Dataset | Multirotor target | Video | Moving airborne camera | Verdict |
| --- | :---: | :---: | :---: | --- |
| **ARD-MAV** | ✅ | ✅ | ✅ | **In use.** Our test set |
| **ARD100** | ✅ | ✅ | ✅ | **Get this next** |
| **MOT-FLY** | ✅ | ✅ | ✅ | **New find — best free addition** |
| **FL-Drones** | ✅ | ✅ | ✅ | Held-out test for M4b; low-res |
| Det-Fly | ✅ | ❌ stills | ✅ | Appearance only |
| MAV-VID | ✅ | ✅ | ⚠️ mixed | Targets too large |
| UAVSwarm | ✅ | ✅ | ⚠️ unverified | Swarm/MOT focus |
| AIRMOT | ✅ | ✅ | ✅ simulated | Synthetic, tiny |
| **NPS-Drones** | ❌ **fixed-wing** | ✅ | ✅ | **Demoted — see below** |
| AOT | ❌ manned aircraft | ✅ | ✅ | Pretraining only |
| MMFW-UAV | ❌ fixed-wing | ✅ | ✅ | Excluded |
| DUT / VisioDECT / Drone-vs-Bird / Anti-UAV* / LRDDv2-3 / MMAUD / MM-UAV | ✅ | mixed | ❌ ground | Auxiliary only |
| SynDroneVision / SimD3 | ✅ | ✅ | ✅ | Synthetic |

---

## ⚠️ Two corrections to earlier notes

### NPS-Drones targets are fixed-wing, not multirotors

I previously recommended NPS-Drones as an easy third option for M4b. **That was wrong for
this project** — the targets are delta-wing fixed-wing UAVs.

Evidence, from the source paper (Li et al., IEEE TETC 2022):

> "The V-shape is likely due to the **delta wing shape of the UAVs**, and the doublet form
> is caused by the motion of the UAVs between frames."

Corroborating:

- The camera was "mounted on a **single UAV flying with a larger group** of multiple UAVs
  flying simultaneously" — the observer and the targets are the same swarm, and the
  observer is explicitly "a custom **delta-wing** airframe."
- Box statistics: max target box is **65×21 px, a 3.1:1 aspect ratio.** A multirotor is
  near-square. Compare FL-Drones at 259×197 (1.3:1) and 9×9.
- The paper's own detector *exploits* the V-shape as a discriminative appearance feature —
  a cue a quadcopter does not produce.

It remains excellent air-to-air data with real ego-motion and a permissive license, so it
is still useful as generic small-object-against-sky pretraining. **It is not a valid
benchmark for multirotor detection**, and a number measured on it does not transfer.

### LRDDv2 is still images, and access is gated

It is distributed as **39,516 annotated still images**, not video — so it fails filter 2
and cannot drive a motion branch. Source clips are not offered.
*(Whether the underlying video exists on request is unverified.)*

Access is also not the open link I implied: the Drexel page routes through a **Google Forms
access request**, with `kp3275@drexel.edu` as the contact. Not a blocker, but not a
one-liner `curl` either.

---

# Tier 1 — passes all three filters

## ARD-MAV — our current test set

**Contents.** 60 videos, 107,497 frames, 1920×1080. Average target is **0.02% of image
area** (~15×10 px). Released with GLAD.

**Targets.** Multirotor. Video filenames are `phantom01`…`phantom100`, indicating **DJI
Phantom** targets. *(The specific model is inferred from the naming; the labels carry only
the class `Drone`.)*

**How it was captured.** A camera-carrying multirotor pursuing a target MAV outdoors,
across three deliberately separated conditions: *ordinary* (clean background), *complex*
(buildings, trees, ground clutter), and *small* (long range). The per-video condition
grouping is published, which is what makes per-category scoring possible.

**Labels.** Pascal VOC XML, one file per frame, single class `Drone`. A missing XML means
*unannotated*, not empty — see [MANIFEST.md](data/processed/ARD-MAV/MANIFEST.md).

**Split.** 45 train/val (random 5:1), 15 test. We use those 15 verbatim.

**Access.** [Google Drive](https://drive.google.com/file/d/1_I5jR-a-Jlan96s7XD3QeLLddb51rDT_/view),
also Baidu (code `z1xb`). **MIT.**

**Verdict.** Already ingested — 28,337 test frames, validated. Weakness is diversity: one
target airframe, one campaign, one lab.

---

## ARD100 — the successor

**Contents.** 100 videos, **202,467 frames**, 1920×1080 @ 30 fps. Supersedes ARD-MAV,
released with YOLOMG.

**Targets.** Multirotor. **The hardest size distribution of any real dataset**: average
target is ~0.01% of frame area, with **42.18% of objects under 12×12 px** and a further
37.55% between 12 and 20 px.

**How it was captured.** Footage shot with **DJI Mavic 2 and DJI M300** cameras — those are
the *observer* platforms, both multirotors. Conditions deliberately include complex
backgrounds, low and strong light, **abrupt camera movement**, and fast-moving targets.

**Access.** Via `github.com/Irisky123/YOLOMG` → Baidu Netdisk, code `1x2z`. Baidu needs an
account and usually the desktop client. Note the YOLOMG *code* is GPL-3.0; the data is
separate.

**Verdict.** The natural M4b "extra" set: exclude by filename any video matching our local
60, and the remainder is unseen. Format-identical to ARD-MAV, so a drop is attributable to
generalisation rather than to our conversion. Weaker independence than FL-Drones (same lab,
likely the same campaign).

---

## MOT-FLY — the best free addition

**New find.** Not in the original survey, and it passes all three filters cleanly.

**Contents.** **16 RGB sequences, 11,186 frames, 31,722 object instances**, all at
1920×1080. Split 8 sequences / 5,535 frames train, 8 sequences / 5,651 frames test.
*(A second source reports the split as 7,238/3,948 — the repo's own numbers are used here.)*

**Targets.** **DJI Phantom 4, DJI Mavic, and two custom-built laboratory UAVs** — all
multirotors, and genuinely *heterogeneous*, which ARD-MAV is not. **One to three targets
per sequence**, so it is multi-target.

**How it was captured.** From **another flying UAV (a DJI Mavic)** — a multirotor observer,
exactly our eventual configuration. Backgrounds span urban, villages, fields and sky, with
varied viewing angles, partial occlusion and variable lighting. **Over 90% of targets
occupy under 5% of image area.**

**Labels.** **MOTChallenge format** — `gt.txt` / `det.txt` per sequence with persistent
track IDs, plus `seqinfo.ini`. That is a tracking schema, so converting to our YOLO layout
means dropping IDs (or keeping them for future track-level evaluation).

**Access.** `github.com/CZC-123/MOT-FLY` — **Google Drive** and Baidu (password `pe53`).
**Apache-2.0**, commercially usable.

**Verdict.** Small (11k frames), but it is free of Baidu friction, permissively licensed,
multi-target, multi-model, and captured from a multirotor. **Best available complement to
ARD-MAV for airframe and multi-target diversity.** The MOTChallenge format also makes it
the natural set for the tracking stage after detection.

---

## FL-Drones

**Contents.** As used in the drone-to-drone literature: **14 videos, 38,948 frames**, at
**640×480 and 752×480**. Target sizes min 9×9, **average 25.5×16.4**, max 259×197.

**Targets.** Multirotor. The paper calls them "mUAVs" without naming an airframe class, but
two facts settle it: the set **contains indoor sequences** (fixed-wing cannot fly indoors),
and the box aspect ratios are near-square, unlike NPS-Drones' 3:1.

**How it was captured.** Rozantsev, Lepetit & Fua (EPFL). "Acquired by **a camera mounted
on a drone filming similar ones** while flying outdoors" — plus indoor sequences. Dogfight
describes it as "quite challenging due to extreme illumination, pose, and size changes …
their shape is **barely retained even in consecutive frames**."

> **Version confusion — read before citing.** Three annotation lineages exist: the EPFL
> original (20 sequences, ~8,000 manual boxes), the 14-video / 38,948-frame benchmark
> version distributed via TransVisDrone prep, and **Dogfight's re-annotation** (their
> Figure 8 shows original vs. corrected boxes with IoU between them, for both FL-Drones and
> NPS-Drones). Always state which you used. *(The exact derivation between them is
> unverified.)*

**Verdict.** The best genuine held-out test for GLAD — GLAD published on ARD-MAV and
NPS-Drones but **not** FL-Drones. Two frictions: access is via another repo's prep script,
and at 752×480 it is a different resolution from GLAD's hardcoded 1080p motion constants,
which must be rescaled first ([glad-model.md](glad-model.md) §6.3).

---

# Tier 2 — fails one filter

## Det-Fly — fails "video"

**Contents.** 13,271 still images at 3840×2160 of a flying target MAV, from another flying
multirotor. **Targets are DJI Mavic 2 class — multirotor.**

**How it was captured.** Systematically balanced so each factor is separable:
background sky / urban / field / mountain (20–30% each); viewing angle front 36.4%, top
32.5%, bottom 31.1%. Nearly half the targets occupy under 5% of image area.

**Labels.** Pascal VOC XML. **Access.** `github.com/Jake-WU/Det-Fly` → OneDrive or Baidu.
**MIT.**

**Verdict.** The best *appearance* and viewpoint diversity in air-to-air — top-down and
bottom-up views against terrain are exactly what a chase produces and what most sets lack.
**But sparse stills with no stated frame ordering**, so motion methods cannot run; it
benchmarks the appearance path alone. ~50 GB at 4K — pull directly onto the GPU instance.

## MAV-VID — fails "targets are small", partially "moving camera"

**Contents.** 64 videos / **40,232 images**; 53 videos (29,500) train, 11 (10,732) val.
Single multirotor target per frame. Name is literally *Multirotor Aerial Vehicle VID*.

**How it was captured.** **Mixed sources** — some from other drones, some from ground-based
surveillance cameras, some handheld mobile devices. So filter 3 holds only for part of it,
and the subset is not cleanly separable.

**The disqualifier:** average object size is **136×77 px (0.66% of image)**, with another
source reporting 215×128 (3.28%). That is **one to two orders of magnitude larger** than
our targets. A model tuned here learns nothing about 10–30 px detection.

**Access.** Kaggle ("multirotor aerial vehicle vid mavvid"), YOLO annotations. License
unstated in the benchmark repo. **Verdict.** Useful only for close-range appearance.

## UAVSwarm — fails "verified viewpoint"

**Contents.** **72 sequences, 12,598 annotated images**, 13 scenarios, **more than 19 UAV
models**, with **3 to 23 UAVs per sequence**. Split 36/36 sequences (6,844 / 5,754 images).
MDPI *Remote Sensing* 2022.

**How it was captured.** Described as diverse camera perspectives with "dynamic camera
movements" and micro-UAV rapid motion. **Whether the footage is airborne or ground-based is
unverified** — the MDPI page returns 403 and secondary sources disagree; SCT-MOT groups it
with air-to-air benchmarks. *(Resolution and provenance also unverified.)*

**Verdict.** By far the widest **airframe variety** (19+ models) and the densest
multi-target scenes. Worth chasing down if swarm/multi-intruder ever becomes a requirement
— which would also break GLAD's single-target design. Verify the viewpoint before using.

## AIRMOT — simulated air-to-air

8 RGB sequences, **7,844 frames** at 1920×1080 (5,124 train / 2,720 test), 5–16 homogeneous
UAVs per frame in varied formations. **Simulated**, so useful for swarm-tracking logic
rather than appearance realism.

## AOT — fails "multirotor target"

**Contents.** 4,943 sequences of ~120 s at 10 Hz. **5.9M+ images, 3.3M+ annotations.**
Largest air-to-air corpus in existence.

**How it was captured.** Amazon Prime Air's detect-and-avoid programme: cameras on an
aircraft recording **planned encounters** (a cooperating intruder on a scripted conflict
trajectory) plus unplanned objects of opportunity. Grayscale, wide FOV, collision-relevant
ranges.

**Labels.** 2D boxes with track IDs, object class, and a **range estimate** — planned
encounters carry GPS from both aircraft, so distance is measured, not annotated.

**Access.** Public S3, no AWS account: `aws s3 ls --no-sign-request
s3://airborne-obj-detection-challenge-training/`. ~13 TB full, ~500 GB partial via the
starter kit's `partial=True`. **CDLA-Permissive-1.0.**

**Verdict.** Most annotated objects are **manned aircraft**. Value here is
small-object-against-sky pretraining at unmatched scale, plus real range labels. Not a
drone benchmark.

## MMFW-UAV — excluded

*Multi-sensor and multi-view **fixed-wing** UAV dataset for air-to-air vision tasks*
(Nature *Scientific Data*, 2025). Correct viewpoint, wrong airframe. Noted so it is not
rediscovered.

---

# Tier 3 — ground-based (auxiliary only)

Useful for airframe variety and bird negatives. **Never** for benchmarking this project.

| Dataset | Contents | Capture | Labels | License |
| --- | --- | --- | --- | --- |
| **DUT Anti-UAV** | 10,000 images + 20 videos | ground/upward cameras, varied outdoor scenes, day & night | boxes, train/val/test provided | **Apache-2.0** |
| **Anti-UAV410** | 410 **thermal IR** videos, 438k boxes, split 200/90/120 | ground thermal cameras "in the wild"; **>half of targets under 50 px** | boxes | research |
| **Anti-UAV** | 318 **paired RGB+IR** sequences, ~580k boxes | ground, registered dual-sensor | boxes | research |
| **VisioDECT** | 20,924 images, **6 airframes** | ground, 30–100 m, 3 weather/lighting scenarios; 20 months, 12+ locations | boxes in `.txt`/`.xml`/`.csv` | IEEE DataPort |
| **Drone-vs-Bird** | 77 videos, **95k+ frames** | ground static & pan-tilt, long range, **real birds in frame** | boxes incl. birds | **DUA, non-commercial** |
| **LRDDv2** | **39,516 still images** @1080p, range on 8k+ | DJI Mavic Air 2 + iPhone/Pixel; targets DJI Mini 3, Mavic Pro; to ~107 m | **YOLO boxes + range in metres** | request form |
| **LRDDv3** | adds **thermal** + range | ICRA 2026 | boxes + range | ICRA 2026 |
| **MMAUD** | ~15 sequences, **rosbags** | stereo + Livox Avia + Mid-360 + mmWave + 4 mic arrays; targets Mavic 2/3, Phantom 4, Avata, M300 | **Leica 3D ground truth**, type + trajectory | **CC BY-NC-SA** |
| **MM-UAV** | 1,321 seq, ~2.8M frames/modality | ground **RGB + IR + event camera**; RGB only 640×360 | boxes + persistent IDs | CC BY-NC-SA |
| **UETT4K** | 4K images | ground, diverse conditions | boxes | IEEE |

**The two worth actually using:**

- **VisioDECT** — the best airframe-diversity source. Six labelled models (Anafi-Extended,
  DJI FPV, DJI Phantom, EFT-E410S, Mavic2-Air, Mavic2-Enterprise) versus one in Det-Fly and
  one in ARD-MAV. If the requirement is "detects drones" rather than "detects this drone",
  this is what proves it.
- **Drone-vs-Bird** — the only real-world source of annotated birds at drone-like apparent
  scale. Bird discrimination is the biggest untested risk in a fielded system. The DUA is
  worth the friction; flag the non-commercial term if that ever matters.

**LRDDv2/v3** remain the only sets labelling **physical distance**, derived from telemetry
rather than eyeballed: ground cameras use `sqrt(horizontal² + height²)` from the target's
flight log; airborne footage uses GPS plus altitude difference via the haversine formula.
Relevant to M6 only if pixel-size → range conversion becomes a requirement.

---

# Tier 4 — synthetic

- **SynDroneVision** — 140,038 rendered images. Open on
  [Zenodo](https://zenodo.org/records/13360116). Augmentation only.
- **SimD3** — **178,639 images at 1080p** (112,899 non-VFX / 46,086 VFX / 19,654 adverse
  weather). Unreal Engine 5 + CoSys-AirSim, **360° six-camera rig** (six synchronised
  virtual cameras, 60° FOV each). **15 drone models, 8 carrying visually distinct payloads**
  (parcels, bags, weapon-shaped), plus **8 annotated bird species**. Labels are pixel-exact
  YOLOv5-format boxes from AirSim's segmentation API, two classes. **Trap:** simulated bird
  *flocks* are deliberately **unannotated**, so they are unlabelled clutter rather than
  negatives. **Not yet released** — "upon acceptance". CC BY 4.0.
- **SynthSwarm** — controllable synthetic UAV *swarm* dataset (JEOS 2026). Relevant only if
  multi-target becomes a requirement.

Synthetic-to-real transfer at 10–30 px is unproven. Treat all of these as pretraining or
augmentation, never validation.

---

# Download recipes

## MOT-FLY (recommended first addition)

`github.com/CZC-123/MOT-FLY` → Google Drive link in the README (or Baidu, password `pe53`).
Apache-2.0. Small enough to stage locally. Arrives in MOTChallenge layout:

```
<seq>/img1/000001.jpg ...
<seq>/gt/gt.txt          # frame, id, x, y, w, h, conf, class, visibility
<seq>/seqinfo.ini
```

Converting to our canonical YOLO layout means dropping the `id` column — keep it in a
sidecar if track-level evaluation is ever wanted.

## NPS-Drones (fixed-wing — pretraining only)

No registration, no Baidu, plain HTTP, **BSD-3-Clause**:

```bash
BASE=https://engineering.purdue.edu/~bouman/UAV_Dataset
curl -O $BASE/Videos.zip                  # 50 videos, 70,250 frames, 30 fps
curl -O $BASE/Video_Annotation-v2.zip     # corrected labels -- USE THIS ONE
curl -O $BASE/Video_Annotation-v1.zip     # original VATIC labels, reference only
curl -O $BASE/pubs/LICENSE.txt
```

Internally the authors call it **U2U-D&TD**, cut from ~100 hours of raw footage; each video
is ~1 minute. A **pitot tube** in the camera's field of view was masked out during
preprocessing. Resolutions are 1920×1080 and **1280×960** (some papers misquote 1280×760).
The original paper used 5-fold cross-validation; the 40/10 split is a later convention.

Cite: Li, Ye, Chung, Kolsch, Wachs & Bouman, *"Fast and Robust UAV to UAV Detection and
Tracking from Video"*, IEEE TETC 10(3):1519-1531, 2022.

## AOT (pretraining scale)

```bash
aws s3 ls --no-sign-request s3://airborne-obj-detection-challenge-training/
```

Pull onto the GPU instance, never here. ~13 TB full; use the starter kit's partial flag.

---

# Traps

- **VisDrone is not a drone-detection dataset.** Cars and pedestrians shot *from* a drone.
  Same for MAVREC, AU-AIR, MOR-UAV, SynDrone — all look *down at the ground*.
  **Air-to-air ≠ aerial.**
- **Check the target airframe.** NPS-Drones (delta-wing) and MMFW-UAV (fixed-wing) are
  air-to-air but not multirotor. AOT is mostly manned aircraft.
- **Check the annotation version.** NPS-Drones ships v1 and v2 from Purdue *and* has a third
  re-annotation from Dogfight. FL-Drones circulates in at least three lineages. Published
  numbers assume specific ones.
- **Split by video, never by frame.** Adjacent frames are near-identical; a random
  frame-level split leaks near-duplicates and inflates mAP by tens of points.
- **Numbers across datasets are not comparable.** YOLOMG's 0.78 AP (ARD100) versus GLAD's
  0.80 (ARD-MAV) says nothing. Neither do runs at different `imgsz` — YOLOMG's own 640→1280
  swing is 0.78→0.85.
- **Target size is a filter too.** MAV-VID is multirotor video but its targets average
  136×77 px. Passing the viewpoint test does not make a dataset relevant.
- **Baidu Netdisk** gates ARD100 and mirrors several others. Frictionless alternatives:
  ARD-MAV, **MOT-FLY**, NPS-Drones, AOT, Anti-UAV410, DUT, SynDroneVision.
- **Licensing, if this goes commercial.** Permissive: **MOT-FLY (Apache-2.0)**, NPS-Drones
  (BSD-3), ARD-MAV and Det-Fly (MIT), DUT (Apache-2.0), AOT (CDLA-Permissive), SimD3
  (CC BY 4.0). Restricted: **Drone-vs-Bird (DUA, non-commercial)**, **MMAUD** and **MM-UAV**
  (CC BY-NC-SA). YOLOMG's *code* is GPL-3.0 even though its data is separate.
