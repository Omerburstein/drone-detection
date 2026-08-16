# Drone detection — dataset & model survey

Carried over from an earlier planning conversation. This is the reference list the
project plan is built on.

> **Shortlist, not the detail.** [datasets.md](datasets.md) has the full entry for every
> dataset below — contents, how it was physically captured, what the labels contain,
> download commands and licensing. [glad-model.md](glad-model.md) does the same for GLAD.

## 1. Datasets

### Air-to-air (camera mounted on a flying drone) — primary match for this use case

| Dataset | Size | Source |
| --- | --- | --- |
| **ARD100** (supersedes ARD-MAV) | 100 videos / 202,467 frames | `github.com/Irisky123/YOLOMG` → Baidu (code `1x2z`) |
| **ARD-MAV** (original 60-video set) | 60 videos / 107,497 frames | GitHub · Google Drive · MIT |
| **NPS-Drones** | 50 videos / 70,250 frames @ 1080p & 1280×760, GoPro-3 on a delta-wing | `engineering.purdue.edu/~bouman/UAV_Dataset/` |
| **Det-Fly** | 13,271 images @ 3840×2160, single DJI Mavic target | `github.com/Jake-WU/Det-Fly` (OneDrive + Baidu links inside) |
| **FL-Drones** | 14 videos / 38,948 frames | via TransVisDrone repo prep instructions |
| **AOT** (Airborne Object Tracking) | 4,943 sequences / 5.9M images / 3.3M boxes | `registry.opendata.aws/airborne-object-tracking/` |

**NPS-Drones** was the notable omission from the first pass. It is the second standard
air-to-air benchmark — GLAD, YOLOMG, Dogfight and TransVisDrone all report on it, so it is
the cheapest way to place a result against published numbers. Targets are tiny (10×8 min,
~16 px average). Standard split is 40 train / 10 test.

**AOT** is by far the largest air-to-air set and the only one with a permissive bulk
download (CDLA-Permissive-1.0, public S3, `aws s3 ls --no-sign-request
s3://airborne-obj-detection-challenge-training/`, no account needed). Two caveats: the full
training set is ~13 TB (a `partial=True` flag gets ~500 GB), and most annotated objects are
**manned aircraft**, not multirotors — it suits detect-and-avoid framing more than our
threat framing. Useful as small-object-against-sky pretraining, not as a drone benchmark.

### Ground-based (multirotor targets) — appearance / airframe diversity

| Dataset | Size | Source |
| --- | --- | --- |
| **DUT Anti-UAV** | 10k images + 20 videos, MIT | `github.com/wangdongdut/DUT-Anti-UAV` |
| **Anti-UAV410** (thermal IR) | 410 videos / 438k boxes | GitHub · Google Drive |
| **Anti-UAV** (RGB+IR pairs) | 318 video pairs, 580k boxes | `github.com/ZhaoJ9014/Anti-UAV` · `anti-uav.github.io` |
| **VisioDECT** | 20,924 images; 6 airframes (Anafi-Extended, DJI FPV, DJI Phantom, EFT-E410S, Mavic2-Air, Mavic2-Enterprise); 3 weather scenarios; 30–100 m | IEEE DataPort + Data in Brief data paper (2026) |
| **Drone-vs-Bird** | 77 videos / 95k+ frames + bird negatives. **DUA required, non-commercial** | `github.com/wosdetc/challenge` |
| **LRDDv2 / LRDDv3** | 39,516 images @ 1920×1080 **with range labels** on 8k+; v3 adds thermal | LRDDv2 `arXiv:2508.03331`, LRDDv3 (ICRA 2026) |
| **MMAUD** | Multi-modal: stereo, LiDAR, radar, audio; Leica ground truth | `github.com/ntu-aris/MMAUD` (ICRA 2024) |
| **SynDroneVision** (synthetic) | 140,038 images | `zenodo.org/records/13360116` |
| **SimD3** (synthetic) | UE5, six-camera rig, payload variation + **bird distractors**, CC-BY-4.0 | `arXiv:2601.14742` |

Newer entries worth knowing about, none of them air-to-air:

- **LRDDv2/v3** are the only sets carrying **measured range per image**. That makes them the
  natural source if we ever need to convert pixel size into a distance estimate for the
  edge budget (M6). Ground-based, so treat as auxiliary.
- **Drone-vs-Bird** was refreshed for the 8th WOSDETC challenge at IJCNN 2025 — still the
  best source of bird negatives, still DUA-gated and non-commercial.
- **SimD3** (2026) is the first synthetic set that models *bird distractors and payload
  variation* deliberately. Synthetic-to-real transfer is unproven for our case, but it is
  free, permissively licensed, and the only cheap source of hard bird negatives without a
  DUA.
- **MMAUD** is ground/overhead multi-sensor. Interesting for sensor fusion later; not a
  match for a monocular camera on a flying drone.

### Access friction

Several ship only via **Baidu Netdisk** (account + client required). The frictionless
ones are **ARD-MAV** (Google Drive), **NPS-Drones** (direct HTTP from Purdue), **AOT**
(unauthenticated S3), **Anti-UAV410** (Google Drive), **DUT Anti-UAV**, and
**SynDroneVision**. Start there if Baidu is a blocker.

For M4b specifically, the ranking is: **FL-Drones** (genuinely held out from GLAD) >
**NPS-Drones** (video, but GLAD published on it, so partly designed-against) >
**ARD100-extra**. NPS-Drones was not considered in the original M4b note because it was
missing from this list; it does not displace FL-Drones, but it is far easier to obtain and
comes with published GLAD numbers to sanity-check against.

### ⚠️ Name trap

**VisDrone is not a drone-detection dataset.** It is cars and pedestrians photographed
*from* a drone. Search engines conflate the two constantly. Ignore VisDrone and every
"VisDrone-pretrained" checkpoint for this project.

## 2. Models

### Purpose-built drone-to-drone detectors (code available)

| Model | Notes | Source |
| --- | --- | --- |
| **GLAD** (Global-Local MAV Detection) | Appearance + motion state machine for tiny targets from a moving camera. Ships pretrained weights. MIT. **Best starting point.** Full write-up: [glad-model.md](glad-model.md) | `github.com/WestlakeIntelligentRobotics/Global-Local-MAV-Detection` |
| **YOLOMG** | YOLOv5-based; fuses RGB with a pixel-level motion-difference map. AP 0.78 @640 / 0.85 @1280 on ARD100, 133/35 FPS. Code + ARD100. GPL-3.0. No pretrained weights — train from `yolov5s.pt`. | `github.com/Irisky123/YOLOMG` |
| **TransVisDrone** (ICRA 2023) | CSPDarkNet-53 + VideoSwin spatio-temporal transformer. AP@0.5: NPS 0.95, FL-Drones 0.75, AOT 0.80. | `github.com/tusharsangam/TransVisDrone` |
| **SiamDT** | Thermal-IR **tracker**, not a detector. Pair with a detector for the chase / lock-on stage. | Anti-UAV410 repo, `trackers/SiamDT/` |

### Off-the-shelf single-class "drone" YOLO weights (ground-based training data)

- `IRIS-Computer-Vision/YOLOv8s_EO_Drone_Detection` — Anti-UAV-derived, tuned for
  long-range / small scale. Closest in spirit to this problem.
- `doguilmak/Drone-Detection-YOLOv11x` — newest architecture, single `drone` class.
- `doguilmak/Drone-Detection-YOLOv7`
- `Javvanny/yolov8m_flying_objects_detection` — drones *and* birds, so it gives bird
  discrimination out of the box.

## 3. Plan

1. **Feasibility check (runs on this laptop).** Pull `YOLOv8s_EO_Drone_Detection` and
   run it over ARD-MAV video. It is expected to do poorly on small/fast targets — that
   is the point. It establishes the baseline number everything else is measured against.
2. **Fine-tune GLAD** from its released weights on ARD100 + Det-Fly, holding out part of
   ARD100. **Requires a rented GPU** — see [hardware.md](hardware.md).
3. **Add diversity:** DUT Anti-UAV + VisioDECT for airframe variety (Det-Fly alone is a
   single Mavic model), and Drone-vs-Bird for bird negatives.
4. GLAD before YOLOMG purely because released weights save the first training run.
   YOLOMG is the stronger published result and is worth trying once the data pipeline
   works.
