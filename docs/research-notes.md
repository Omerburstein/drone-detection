# Drone detection — dataset & model survey

Carried over from an earlier planning conversation. This is the reference list the
project plan is built on.

## 1. Datasets

### Air-to-air (camera mounted on a flying drone) — primary match for this use case

| Dataset | Size | Source |
| --- | --- | --- |
| **ARD100** (supersedes ARD-MAV) | 100 videos / 202,467 frames | `github.com/Irisky123/YOLOMG` → Baidu (code `1x2z`) |
| **ARD-MAV** (original 60-video set) | 60 videos / 107,497 frames | GitHub · Google Drive · MIT |
| **Det-Fly** | 13,271 images @ 3840×2160, single DJI Mavic target | `github.com/Jake-WU/Det-Fly` (OneDrive + Baidu links inside) |
| **FL-Drones** | 14 videos / 38,948 frames | via TransVisDrone repo prep instructions |

### Ground-based (multirotor targets) — appearance / airframe diversity

| Dataset | Size | Source |
| --- | --- | --- |
| **DUT Anti-UAV** | 10k images + 20 videos, MIT | `github.com/wangdongdut/DUT-Anti-UAV` |
| **Anti-UAV410** (thermal IR) | 410 videos / 438k boxes | GitHub · Google Drive |
| **Anti-UAV** (RGB+IR pairs) | 318 video pairs, 580k boxes | `github.com/ZhaoJ9014/Anti-UAV` · `anti-uav.github.io` |
| **VisioDECT** | 20,924 images; 6 airframes (Anafi-Extended, DJI FPV, DJI Phantom, EFT-E410S, Mavic2-Air, Mavic2-Enterprise); 3 weather scenarios; 30–100 m | IEEE DataPort + Data in Brief data paper (2026) |
| **Drone-vs-Bird** | 77 sequences + bird negatives. **DUA required, non-commercial** | `github.com/wosdetc/challenge` |
| **SynDroneVision** (synthetic) | 140,038 images | `zenodo.org/records/13360116` |

### Access friction

Several ship only via **Baidu Netdisk** (account + client required). The frictionless
ones are **ARD-MAV** (Google Drive), **Anti-UAV410** (Google Drive), **DUT Anti-UAV**,
and **SynDroneVision**. Start there if Baidu is a blocker.

### ⚠️ Name trap

**VisDrone is not a drone-detection dataset.** It is cars and pedestrians photographed
*from* a drone. Search engines conflate the two constantly. Ignore VisDrone and every
"VisDrone-pretrained" checkpoint for this project.

## 2. Models

### Purpose-built drone-to-drone detectors (code available)

| Model | Notes | Source |
| --- | --- | --- |
| **GLAD** (Global-Local MAV Detection) | Appearance + motion, built for tiny targets from a moving camera. Ships pretrained weights in `/weights`. MIT. **Best starting point.** | `github.com/WindyLab/Global-Local-MAV-Detection` |
| **YOLOMG** | YOLOv5-based; fuses RGB with a pixel-level motion-difference map. Code + ARD100. GPL-3.0. No pretrained weights — train from `yolov5s.pt`. | `github.com/Irisky123/YOLOMG` |
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
