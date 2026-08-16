# Dataset reference

Every drone dataset considered for this project: what is in it, how it was physically
captured, what the labels actually say, how to get it, and whether it is useful here.

[research-notes.md](research-notes.md) is the shortlist and the plan. **This file is the
detail.** Facts I could not confirm from a primary source are marked *(unverified)* rather
than guessed.

---

## The distinction that decides everything

**Where was the camera?** This project is air-to-air: camera on a *flying* drone, target
another drone, both moving, background swinging between sky and ground clutter.

| Viewpoint | What it looks like | Use for us |
| --- | --- | --- |
| **Air-to-air** | Camera on a UAV. Background alternates sky/terrain, ego-motion is violent, targets 10–30 px | **Train and test on this** |
| **Ground-to-air** | Camera on a tripod looking up. Mostly sky background, static camera | Airframe variety, bird negatives. Do **not** benchmark on it |
| **Synthetic** | Rendered | Free hard negatives; transfer unproven |

A ground-based set is not a substitute. Static camera means frame differencing works
trivially, and a sky-only background removes the clutter that actually causes our false
positives. A model validated there will not survive M3's finding that precision collapses
on complex backgrounds.

---

## Quick comparison

| Dataset | Viewpoint | Size | Modality | Labels | License | Access |
| --- | --- | --- | --- | --- | --- | --- |
| **ARD-MAV** | air-to-air | 60 vid / 107,497 fr | RGB 1080p | VOC XML boxes | MIT | Google Drive ✅ |
| **ARD100** | air-to-air | 100 vid / 202,467 fr | RGB 1080p | boxes | GPL-3.0 (repo) | Baidu ⚠️ |
| **NPS-Drones** | air-to-air | 50 vid / 70,250 fr | RGB 1080p/1280×760 | boxes, up to 8 targets | **BSD-3** | direct HTTP ✅ |
| **Det-Fly** | air-to-air | 13,271 img | RGB 4K | VOC XML boxes | MIT | OneDrive/Baidu ⚠️ |
| **FL-Drones** | air-to-air | 14 vid / 38,948 fr | RGB 752×480 | boxes | research | via TransVisDrone ⚠️ |
| **AOT** | air-to-air | 4,943 seq / 5.9M img | grayscale | 3.3M boxes + range | CDLA-Perm | open S3 ✅ (13 TB) |
| **DUT Anti-UAV** | ground | 10k img + 20 vid | RGB | boxes | Apache-2.0 | Google Drive ✅ |
| **Anti-UAV410** | ground | 410 vid / 438k box | **thermal IR** | boxes | research | Google Drive ✅ |
| **Anti-UAV** | ground | 318 pairs / 580k box | **RGB + IR** | boxes | research | GitHub ✅ |
| **VisioDECT** | ground | 20,924 img | RGB | boxes, 6 airframes | IEEE DataPort | DataPort ✅ |
| **Drone-vs-Bird** | ground | 77 vid / 95k+ fr | RGB | boxes + **birds** | **DUA, non-comm** | signed request ⚠️ |
| **LRDDv2** | ground | 39,516 img | RGB 1080p | boxes + **range (m)** | see paper | Drexel ✅ |
| **LRDDv3** | ground | — | RGB + **thermal** | boxes + range | see paper | ICRA 2026 ⚠️ |
| **MMAUD** | ground/overhead | ~15 seq | **stereo+LiDAR+radar+audio** | **3D pose** | CC BY-NC-SA | OneDrive/Drive ✅ |
| **SynDroneVision** | synthetic | 140,038 img | RGB | boxes | open | Zenodo ✅ |
| **SimD3** | synthetic | 178,639 img | RGB 1080p | boxes, drone+bird | CC BY 4.0 | **not yet released** ❌ |

---

# Air-to-air datasets

## ARD-MAV — our current test set

**Contents.** 60 videos, 107,497 frames, 1920×1080. Average target is **0.02% of image
area** — that is roughly 15×10 px in a 1080p frame. Released with GLAD.

**How it was captured.** A camera-carrying multirotor pursuing a target MAV outdoors,
across three deliberately separated difficulty conditions: *ordinary* (clean background),
*complex* (buildings, trees, ground clutter), and *small* (long range). The condition
grouping is published per video, which is what makes per-category scoring possible.

**Labels.** Pascal VOC XML, one file per frame, single class `Drone`. A missing XML means
*unannotated*, not empty — see [MANIFEST.md](data/processed/ARD-MAV/MANIFEST.md).

**Split.** 45 train/val (random 5:1), 15 test. We use those 15 verbatim.

**Access.** [Google Drive](https://drive.google.com/file/d/1_I5jR-a-Jlan96s7XD3QeLLddb51rDT_/view),
also Baidu (code `z1xb`). MIT.

**Verdict.** Already ingested — 28,337 test frames, validated. Its weakness is diversity:
one target airframe, one capture campaign, one lab.

---

## ARD100 — the successor

**Contents.** 100 videos, 202,467 frames. Supersedes ARD-MAV and released with YOLOMG.
Harder: YOLOMG scores 0.78 AP here versus GLAD's 0.80 on the 60-video ARD-MAV, and those
two numbers are **not comparable** (different data, different split).

**Access.** Via `github.com/Irisky123/YOLOMG` → Baidu Netdisk, code `1x2z`. Baidu needs an
account and usually the desktop client.

**Verdict.** The natural M4b "extra" set: exclude by filename any video matching our local
60, and whatever remains is unseen. Format-identical to ARD-MAV, so a performance drop is
attributable to generalisation rather than to our conversion. Weaker independence than
FL-Drones (same lab, likely the same capture campaign).

---

## NPS-Drones — the other standard benchmark

**Contents.** 50 videos, **70,250 frames**, 30 fps, 1920×1080 and 1280×760. Targets are
tiny: min 10×8 px, **average ~16 px**, max 65×21. Crucially, **up to 8 target UAVs appear
per sequence** — it is genuinely multi-target, unlike ARD-MAV.

**How it was captured.** Naval Postgraduate School. **GoPro-3 cameras mounted on a custom
delta-wing airframe** — a fixed-wing chase aircraft, not a multirotor. That gives smoother,
faster ego-motion than ARD-MAV's quadrotor footage, and the targets are various small UAVs
of differing appearance.

**Labels.** Bounding boxes originally annotated with VATIC. **Two annotation versions
ship** — v1 (original) and v2 (corrected). The literature standard is **v2**; papers
citing "the clean version" mean this. Use v2 or your numbers will not line up.

**Split.** Standard is 40 train/val, 10 test.

**Why it matters.** GLAD, YOLOMG, Dogfight and TransVisDrone all publish on it, so it is
the cheapest way to place a result against published numbers. **BSD-3-Clause** — the most
permissive license of any real air-to-air set, and commercially usable.

### How to download it

No registration, no Baidu, plain HTTP from Purdue. Three archives plus the license:

```bash
BASE=https://engineering.purdue.edu/~bouman/UAV_Dataset

curl -O $BASE/Videos.zip                  # 50 videos, 70,250 frames
curl -O $BASE/Video_Annotation-v2.zip     # corrected labels -- USE THIS ONE
curl -O $BASE/Video_Annotation-v1.zip     # original VATIC labels, for reference only
curl -O $BASE/pubs/LICENSE.txt            # BSD 3-Clause
```

PowerShell equivalent:

```powershell
$base = 'https://engineering.purdue.edu/~bouman/UAV_Dataset'
foreach ($f in 'Videos.zip','Video_Annotation-v2.zip') {
    Invoke-WebRequest "$base/$f" -OutFile "data/raw/NPS-Drones/$f"
}
```

Pull it **onto the GPU instance**, not here, if it is destined for training. Sizes are not
published on the page; budget for tens of GB on the video archive.

Cite: Li, Ye, Chung, Kolsch, Wachs & Bouman, *"Fast and Robust UAV to UAV Detection and
Tracking from Video"*, IEEE TETC 10(3):1519-1531, 2022. Reference implementation:
`github.com/jingliinpurdue/Fast-and-Robust-UAV-to-UAV-Detection-and-Tracking`.

**Caveat for us.** GLAD published on NPS-Drones, so it is partly designed-against — it is a
weaker independence test than FL-Drones. Also, its multi-target nature breaks GLAD's
single-target state machine, which tracks exactly one `init_rect`.

---

## Det-Fly

**Contents.** 13,271 still images at 3840×2160 of a flying target MAV, taken **from another
flying multirotor (DJI Mavic 2 class)**. Deliberately balanced across conditions rather
than being continuous video.

**How it was captured.** Systematically varied so each factor is separable:

- Background: sky, urban, field, mountain (roughly 20–30% each)
- Viewing angle: front 36.4%, top 32.5%, bottom 31.1% — top-down and bottom-up views
  against terrain are exactly what a chase scenario produces and what most sets lack
- Nearly half the targets occupy **under 5% of image area**

**Labels.** Pascal VOC XML boxes, single target class.

**Access.** `github.com/Jake-WU/Det-Fly` → OneDrive or Baidu. MIT.

**Verdict.** The best *appearance* diversity for air-to-air, and the only one with
deliberate viewpoint balance. **But it is sparse stills with no stated frame ordering**, so
any motion-based method (GLAD, YOLOMG) cannot run on it — it benchmarks the appearance path
alone. That is why M4b rejected it. ~50 GB at 4K; pull it directly onto the GPU instance.

---

## FL-Drones

**Contents.** As used in the drone-to-drone literature (TransVisDrone, Dogfight): **14
videos, 38,948 frames**, low resolution — the source EPFL data is 752×480.

**How it was captured.** Originates from Rozantsev, Lepetit & Fua, *"Flying Objects
Detection from a Single Moving Camera"* (EPFL, CVPR 2015). A camera mounted on a drone
flying **indoors and outdoors**, with a second UAV as target. The original paper describes
20 sequences at 752×480 with ~8,000 annotated boxes.

> **Count discrepancy — read before citing.** The original EPFL release and the "FL-Drones"
> used by the detection literature do not match (20 videos vs 14; 8k boxes vs 38,948
> frames). The benchmark version is a re-annotated, re-segmented subset distributed through
> the TransVisDrone prep instructions. **Always state which version you used.** *(The exact
> relationship between the two is unverified.)*

**Verdict.** The best genuine held-out test for GLAD — GLAD published on ARD-MAV and
NPS-Drones but **not** FL-Drones. Two frictions: access is via another repo's prep script,
and at 752×480 it is a different resolution than GLAD's hardcoded 1080p motion constants,
so those must be rescaled first ([glad-model.md](glad-model.md) §6.3).

---

## AOT — Airborne Object Tracking

**Contents.** 4,943 flight sequences of ~120 s each at 10 Hz. **5.9M+ images, 3.3M+ 2D
annotations.** By a wide margin the largest air-to-air set in existence.

**How it was captured.** Amazon Prime Air's detect-and-avoid programme. Cameras on an
aircraft recording **planned encounters** (a cooperating intruder flown on a scripted
conflict trajectory) plus **unplanned airborne objects** of opportunity. Grayscale, wide
field of view, targets at genuine collision-relevant ranges.

**Labels.** 2D bounding boxes with track IDs, object class, and a **range/distance
estimate** — planned encounters carry GPS from both aircraft, so distance is real
measurement, not annotation guesswork.

**Access.** Public S3, **no AWS account required**:

```bash
aws s3 ls --no-sign-request s3://airborne-obj-detection-challenge-training/
```

Full training set is **~13 TB**; the starter kit exposes a `partial=True` flag for a ~500 GB
subset. License is **CDLA-Permissive-1.0** — commercially usable.

**Verdict.** The catch: most annotated objects are **manned aircraft** (planes,
helicopters), not multirotors. It is a *detect-and-avoid* corpus, not a counter-drone one.
Its value here is as **small-object-against-sky pretraining** at a scale nothing else
offers, plus it is the only air-to-air source with real range labels. Do not report it as
a drone-detection benchmark.

---

# Ground-based datasets

Useful for airframe variety and bird negatives. Never for benchmarking this project.

## DUT Anti-UAV

**Contents.** A detection set of **10,000 images** plus a tracking set of **20 videos**
(short- and long-term sequences). All manually annotated. From *"Vision-based Anti-UAV
Detection and Tracking"*, IEEE T-ITS 2022.

**How it was captured.** Ground-level and upward-looking cameras across varied outdoor
scenes, day and night, with sky, buildings and vegetation backgrounds. *(Per-split counts,
resolutions and capture hardware are not stated in the repo README or abstract —
unverified; the full paper has them.)*

**Access.** `github.com/wangdongdut/DUT-Anti-UAV`, Google Drive **and** Baidu, train/val/
test provided separately. **Apache-2.0** — commercially usable.

**Verdict.** Frictionless and permissive. Good airframe/appearance variety to mix into
fine-tuning; useless as a benchmark for us.

## Anti-UAV410

**Contents.** **410 thermal-infrared videos**, 438k+ manually annotated boxes. Split
200 train / 90 val / 120 test. **Over half the targets are under 50 px.**

**How it was captured.** Ground-based thermal IR cameras tracking drones "in the wild"
across day/night and varied backgrounds. From TPAMI 2023. *(Resolution and camera model
unverified.)*

**Access.** `github.com/HwangBo94/Anti-UAV410`, Google Drive or Baidu (code `a410`).

**Verdict.** Thermal, so it does not transfer to our RGB pipeline. Relevant only if the
sensor plan ever adds an IR channel. Its size distribution is the closest to ours of any
ground set.

## Anti-UAV (RGB+IR)

**Contents.** 318 **paired** RGB and infrared video sequences, ~580k boxes. The
multi-modal registration is the point.

**Access.** `github.com/ZhaoJ9014/Anti-UAV`, `anti-uav.github.io`.

**Verdict.** Only interesting for a future RGB+IR fusion sensor.

## VisioDECT

**Contents.** 20,924 images across **six distinct airframes** — Anafi-Extended, DJI FPV,
DJI Phantom, EFT-E410S, Mavic2-Air, Mavic2-Enterprise — at **30–100 m** range, under three
weather/lighting scenarios (sunny, cloudy, evening).

**How it was captured.** Ground-based cameras, each drone flown separately so the airframe
label is exact.

**Access.** IEEE DataPort, with a *Data in Brief* data paper.

**Verdict.** **The best airframe-diversity source available.** Det-Fly is one Mavic and
ARD-MAV is one target; VisioDECT is six labelled models. If the requirement is "detects
drones", not "detects this drone", this is the set that proves it.

## Drone-vs-Bird

**Contents.** 77 videos, **95k+ frames**, refreshed for the 8th WOSDETC challenge at
**IJCNN 2025**. The distinguishing feature: **birds appear and are annotated**, so it is
the only real-data source of the hard negative that matters most.

**How it was captured.** Ground-based static and pan-tilt cameras at long range, with
genuine birds in frame at similar apparent scale to the drones.

**Access.** `github.com/wosdetc/challenge`. **Requires a signed Data Usage Agreement and
is non-commercial.** Flag this the moment a commercial use case appears.

**Verdict.** Bird discrimination is the single biggest untested risk in a fielded
counter-drone system — a distant bird and a distant quadcopter are a handful of pixels
each. This is the only real-world set that addresses it. The DUA is worth the friction.

## LRDDv2 / LRDDv3 — the range-labelled sets

**LRDDv2 contents.** **39,516 annotated images at 1920×1080**, with **measured range for
over 8,000** of them. The majority of images show drones occupying **50 px or fewer** —
a size distribution close to ours.

**How it was captured.** Deliberately mixed platforms:

- **Cameras:** a DJI Mavic Air 2 (1080p/30, so part of the set *is* airborne), plus
  iPhone 12 / 15 Pro Max and Google Pixel 6 handhelds
- **Targets:** DJI Mini 3 and DJI Mavic Pro
- **Range:** out to **350 feet (~107 m)**
- **Conditions:** backgrounds labelled City / Grass / Sky / Water; weather labelled Sunny /
  Clear / Cloudy / Rainy; both camera and target in motion for part of the set

**Labels.** **YOLO format** boxes (manual labelling plus YOLOv5-assisted pre-annotation),
plus the range value. Range is not eyeballed — it is derived from telemetry:

- ground camera: `distance = sqrt(horizontal² + height²)` from the drone's own flight log
- airborne camera: GPS coordinates plus altitude difference via the haversine formula

**LRDDv3** (ICRA 2026) extends this with **thermal imagery** alongside RGB, keeping range.

**Access.** `research.coe.drexel.edu/ece/imaple/lrddv2/`.

**Verdict.** The **only sets that label physical distance**, which makes them the natural
source if M6's edge budget ever needs pixel-size → range conversion (how far away can we
detect, and how much closing time does that buy). Mostly ground-based, and background
labels are coarse, so treat as auxiliary rather than training core.

## MMAUD — multi-sensor

**Contents.** Flight sequences at three difficulty tiers: V1 rooftop/simple (below 30 m),
V2 carpark/hard and V3 carpark/moderate (up to 100 m). ICRA 2024.

**How it was captured.** Not a camera dataset — a **sensor rig**:

| Modality | Hardware |
| --- | --- |
| Vision | two time-synchronised cameras (stereo) |
| LiDAR | Livox Avia (conic) + Livox Mid-360 (peripheral) |
| Radar | one mmWave unit |
| Audio | four microphone array nodes |

Targets are **DJI Mavic 2, Mavic 3, Phantom 4, Avata and M300** — good airframe spread.
Ambient heavy-machinery noise was deliberately included so the audio task is not trivial.
The rig looks **upward/overhead** from ground level rather than from a specific vantage.

**Labels.** **Leica-generated ground truth** — a laser tracker giving true 3D position,
which is far more accurate than hand-drawn boxes. Supports three tasks: detection, **UAV
type classification**, and **trajectory estimation**. Distributed as **ROS bags**, not
images.

**Access.** `github.com/ntu-aris/MMAUD`, OneDrive and Google Drive. V1 sequences run
181–428 s at **11.1–19.7 GB each** (rosbag-compressed, ~3× smaller than raw).
**CC BY-NC-SA 4.0 — non-commercial.**

**Verdict.** Wrong viewpoint and wrong format for us — rosbags of a ground sensor rig, not
air-to-air video. Worth knowing for two reasons: it is the reference for **UAV-type
classification** if the requirement grows beyond "is it a drone", and its Leica ground
truth is the accuracy standard for trajectory work. Non-commercial license.

---

# Synthetic datasets

## SynDroneVision

**Contents.** 140,038 rendered images. **Access.** `zenodo.org/records/13360116` — open,
frictionless. **Verdict.** Large and free; use as augmentation, never as validation.

## SimD3

**Contents.** **178,639 images at 1920×1080**, in three subsets: non-VFX (112,899), VFX
(46,086), and adverse weather (19,654).

**How it was generated.** Unreal Engine 5 with the **CoSys-AirSim** plugin. A **360°
six-camera rig** — six synchronised virtual cameras, 60° horizontal FOV each, uniformly
spaced in azimuth for full panoramic coverage. Seven marketplace environments (City Park,
City Creator, Downtown, Dynamic City, Rural Australia, Bridge, Wild West Town), with time
of day, sun angle, cloud cover, fog, rain and snow varied from light haze to heavy.

**What makes it interesting.** Two things nothing else offers together:

- **15 drone models**, of which **8 carry visually distinct payloads** — parcels, bags, and
  weapon-shaped objects. That is a threat-classification signal no real dataset has.
- **8 bird species as annotated distractors**, plus Niagara-simulated bird *flocks* which
  are deliberately left **unannotated** (so flocks act as unlabelled clutter, not negatives
  — worth knowing before training on it).

**Labels.** **YOLOv5-format** normalised boxes, two classes (`drone`, `bird`), generated
automatically from AirSim's segmentation API — so they are pixel-exact, with no annotator
error.

**Access.** **Not yet available.** The paper states release "upon acceptance". CC BY 4.0.

**Verdict.** The only free, permissive, DUA-free source of bird negatives, and the only
source of payload variation. Synthetic-to-real transfer for 10–30 px targets is unproven,
so treat it as pretraining or augmentation. Recheck availability before planning around it.

---

# Traps

- **VisDrone is not a drone-detection dataset.** Cars and pedestrians shot *from* a drone.
  Search engines conflate it constantly. Ignore it and every "VisDrone-pretrained" weight.
- **Split by video, never by frame.** Adjacent frames are near-identical; a random
  frame-level split leaks near-duplicates into validation and inflates mAP by tens of
  points.
- **Numbers across datasets are not comparable.** YOLOMG's 0.78 AP (ARD100) versus GLAD's
  0.80 (ARD-MAV) says nothing. Neither do runs at different `imgsz` — resolution alone
  moves mAP by tens of points on small targets.
- **Check the annotation version.** NPS-Drones ships v1 and v2 labels; FL-Drones circulates
  in at least two segmentations. Published numbers assume specific ones.
- **Baidu Netdisk** gates ARD100 and mirrors several others: account plus, usually, the
  desktop client. Frictionless alternatives: ARD-MAV, **NPS-Drones**, **AOT**,
  Anti-UAV410, DUT Anti-UAV, LRDDv2, SynDroneVision.
- **Licensing, if this ever goes commercial.** Permissive: **NPS-Drones (BSD-3)**, ARD-MAV
  and Det-Fly (MIT), DUT (Apache-2.0), AOT (CDLA-Permissive), SimD3 (CC BY 4.0).
  Restricted: **Drone-vs-Bird (DUA, non-commercial)**, **MMAUD (CC BY-NC-SA)**, and YOLOMG's
  code is GPL-3.0 even though the data is separate.
- **Air-to-air ≠ aerial.** Most "UAV datasets" are captured *from* drones looking *down* at
  ground objects. Only the six sets in the first section point a camera at another aircraft.
