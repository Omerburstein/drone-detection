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
| 2 | **Contiguous frames, not sparse stills** | Every method that works at this target size (GLAD, YOLOMG, Dogfight, TransVisDrone) uses inter-frame motion. Sparse stills cannot drive a motion branch at all. |
| 3 | **Camera on a moving airborne platform** | A static camera makes frame differencing trivial and removes the ego-motion that motion compensation exists to cancel. Ground footage also lacks the sky/terrain background swing that causes our false positives. |

**Only four real datasets pass all three.** They are in Tier 1 below.

### "Video" here means contiguous frames, not a container format

Most of these datasets ship as **extracted numbered frames** (`000001.jpg, 000002.jpg …`)
rather than MP4 files. **That is fine** — arguably better, since it skips a decode and any
re-encoding artefacts. What a motion branch needs is *ordered consecutive frames at a known
rate*, not a video container.

The real dividing line is therefore:

| | Example | Motion branch |
| --- | --- | --- |
| **MP4 / video files** | ARD-MAV, ARD100, Anti-UAV300 | ✅ works |
| **Extracted frame sequences** | MOT-FLY, DUT *tracking*, Anti-UAV410 | ✅ works |
| **Sparse independent stills** | Det-Fly, VisioDECT, LRDDv2, UETT4K, DUT *detection* | ❌ cannot run |

So "DUT Anti-UAV is pictures" is both right and incomplete: it ships **two separate
subsets**, and only one of them is usable —

- **detection subset** — 10,000 *independent* images. Sparse stills. Motion methods cannot
  use it.
- **tracking subset** — 20 sequences delivered as an **Images** archive plus a **Ground
  Truth** archive, i.e. contiguous extracted frames. Usable.

Same for **Drone-vs-Bird**: its annotations are frame-indexed
(`framenum num_objs obj1_x obj1_y obj1_w obj1_h obj1_class …`), which only makes sense over
contiguous sequences. What arrives after the DUA is the sequence data, not a shuffled image
set. *(Exact container unverified — it is only visible post-DUA.)*

---

## Fit summary

| Dataset | Multirotor target | Video | Moving airborne camera | Verdict |
| --- | :---: | :---: | :---: | --- |
| **ARD-MAV** | ✅ | ✅ | ✅ | **In use.** Our test set |
| **ARD100** | ✅ | ✅ | ✅ | **Get this next — now open on Zenodo** |
| **MOT-FLY** | ✅ | ✅ | ✅ | ❌ **download dead since 2024** |
| **FL-Drones** | ✅ | ✅ | ✅ | Held-out test for M4b; low-res |
| Det-Fly | ✅ | ❌ stills | ✅ | Appearance only |
| MAV-VID | ✅ | ✅ | ⚠️ mixed | Targets ~33× too large; terminal-phase only |
| AIRMOT | ✅ | ✅ | ✅ simulated | Synthetic, tiny |
| **NPS-Drones** | ❌ **fixed-wing** | ✅ | ✅ | **Demoted — pretraining only** |
| AOT | ❌ manned aircraft | ✅ | ✅ | Pretraining only |
| **UAVSwarm** | ✅ | ✅ | ❌ **ground-to-air + air-to-ground** | **Ruled out — verified** |
| MMFW-UAV | ❌ fixed-wing | ✅ | ✅ | Excluded |
| DUT / VisioDECT / Drone-vs-Bird / Anti-UAV* / LRDDv2-3 / MMAUD / MM-UAV / USC-Drone | ✅ | mixed | ❌ ground | Auxiliary only |
| SynDroneVision / SimD3 | ✅ | ✅ | ✅ | Synthetic |

---

# Download index

Every direct URL, verified against the source repo/page on 2026-08-17 unless marked.
**Status** is whether the link resolves, not whether the download completes.

## Air-to-air

| Dataset | Host | URL | Status |
| --- | --- | --- | --- |
| **ARD-MAV** | Google Drive | `https://drive.google.com/file/d/1_I5jR-a-Jlan96s7XD3QeLLddb51rDT_/view` | ✅ in use |
| ARD-MAV | Baidu | `https://pan.baidu.com/s/1SmbyjC0l6uye_ghWhEErsQ` code `z1xb` | ✅ |
| **ARD100** ⭐ | **Zenodo** | `https://zenodo.org/records/15870538/files/ARD100.zip?download=1` | ✅ **open, 27.35 GB, CC-BY-4.0** |
| ARD100 | Baidu (original) | `https://pan.baidu.com/s/1ycAoKbzQ1rlzvKr8VRakgw?pwd=1x2z` code `1x2z` | ✅ but throttled |
| **NPS-Drones** | Purdue HTTP | `https://engineering.purdue.edu/~bouman/UAV_Dataset/Videos.zip` | ✅ ⚠️ fixed-wing targets |
| NPS-Drones labels | Purdue HTTP | `…/UAV_Dataset/Video_Annotation-v2.zip` (use v2) | ✅ |
| **MOT-FLY** | Google Drive | `https://drive.google.com/file/d/1GiWLF8B18FGDcCSuSuvGokczCkP_NEgo/view` | ❌ **dead** |
| MOT-FLY | Baidu | `https://pan.baidu.com/s/1eS84Ooz0URojz1tAJNZ5Eg?pwd=pe53` code `pe53` | ❓ untested |
| **Det-Fly** | repo → OneDrive/Baidu | `https://github.com/Jake-WU/Det-Fly` | ✅ sparse stills |
| **FL-Drones** imagery | EPFL CVLab Drive | `https://drive.google.com/open?id=18CoTpjMs80dfanYNpbznjL4e-KB_Diel` | ✅ **no Baidu** |
| **FL-Drones** labels | Dogfight repo (git) | `https://github.com/mwaseema/Drone-Detection/tree/main/annotations/FL-Drones-Dataset` | ✅ 14 files |
| **NPS-Drones** labels (alt) | Dogfight repo (git) | `…/annotations/NPS-Drones-Dataset` | ✅ 50 files, re-annotated |
| **AOT** | open S3 | `aws s3 ls --no-sign-request s3://airborne-obj-detection-challenge-training/` | ✅ 13 TB |

## Ground-based

| Dataset | Host | URL | Format |
| --- | --- | --- | --- |
| **Anti-UAV300** ⭐ | Google Drive | `https://drive.google.com/file/d/1NPYaop35ocVTYWHOYQQHn8YHsM9jmLGr/view` | **RGB + IR video**, Full HD |
| Anti-UAV300 | Baidu | `https://pan.baidu.com/s/1dJR0VKyLyiXBNB_qfa2ZrA` code `sagx` | ” |
| **Anti-UAV410** | Google Drive | `https://drive.google.com/file/d/1zsdazmKS3mHaEZWS2BnqbYHPEcIaH5WR/view` | **IR only**, train/val/test dirs |
| Anti-UAV410 | Baidu | `https://pan.baidu.com/s/1R-L9gKIRowMgjjt52n48-g?pwd=a410` code `a410` | ” |
| Anti-UAV410 results | Google Drive | `https://drive.google.com/file/d/1AlLpoMorj-7bKA1zqo1DkuEZ9h0jQs_-/view` | tracking outputs |
| **Anti-UAV600** | ModelScope | `https://modelscope.cn/datasets/ly261666/3rd_Anti-UAV/files` | IR only |
| **DUT** detection train | Google Drive | `https://drive.google.com/open?id=1RVsSGPUKTdmoyoPTBTWwroyulLek1eTj` | sparse stills |
| DUT detection val | Google Drive | `https://drive.google.com/open?id=1333uEQfGuqTKslRkkeLSCxylh6AQ0X6n` | sparse stills |
| DUT detection test | Google Drive | `https://drive.google.com/open?id=1L1zeW1EMDLlXHClSDcCjl3rs_A6sVai0` | sparse stills |
| **DUT tracking images** ⭐ | Google Drive | `https://drive.google.com/open?id=1dlSPDggg6TRFMcC1jlYIJxxzUQS1mIh9` | **frame sequences** |
| DUT tracking GT | Google Drive | `https://drive.google.com/open?id=16PE3tBhT0lUGZLA8-zIRYvNUvxfhFZJq` | labels for the above |
| DUT (Baidu mirrors) | Baidu | det: `1-ogC7P_K6lwYAqIS8bgIUQ` `u955` · `15sekmPn0hYNQS05Makbmtw` `wkzs` · `1GiA-bKlvMSBkzUwYvo-RiA` `ik4d` · trk: `1OTExqKgvUnqpENtTDu_gGQ` `oine` · `1nkGNERDVgmYIAiwFTdj2xA` `e8mr` | prefix `https://pan.baidu.com/s/` |
| **Drone-vs-Bird** | email DUA | `wosdetc@googlegroups.com` · repo `https://github.com/wosdetc/challenge` | frame-indexed annos |
| **LRDDv2** | request form | `https://research.coe.drexel.edu/ece/imaple/lrddv2/` · `kp3275@drexel.edu` | **stills** |
| **VisioDECT** | IEEE DataPort | search "VisioDECT" | **stills** |
| **MMAUD** | OneDrive/Drive | `https://github.com/ntu-aris/MMAUD` | **rosbags** |
| **SynDroneVision** | Zenodo | `https://zenodo.org/records/13360116` | synthetic stills |

⭐ = the two worth grabbing first for ground-shot multirotor sequences.

**Google Drive `open?id=` links** are the old form; they still resolve. To fetch on a
headless GPU instance, the id is the part after `id=`:

```bash
py -3.13 -m pip install gdown
gdown 1dlSPDggg6TRFMcC1jlYIJxxzUQS1mIh9 -O dut_tracking_images.zip
gdown 16PE3tBhT0lUGZLA8-zIRYvNUvxfhFZJq -O dut_tracking_gt.zip
```

If a Drive link reports quota exceeded (as opposed to "does not exist"), open it in a
browser, **Add shortcut to Drive**, and download from your own copy — your copy carries its
own quota.

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

**Access.** ✅ **Held locally since 2026-08-19.** Zenodo, DOI `10.5281/zenodo.15870538`,
27.35 GB, CC-BY-4.0, MD5 verified — no Baidu needed. The original route (`github.com/
Irisky123/YOLOMG` → Baidu Netdisk, code `1x2z`) needs an account and is throttled to
roughly 100 KB/s. Note the YOLOMG *code* is GPL-3.0; the data is separate. Full download
and archive audit in [PROVENANCE.md](data/raw/ARD100/PROVENANCE.md).

**Prepared.** 15 videos — ARD100's test split ∩ "not in our local 60" — extracted to
`data/raw/ARD100/{videos,Annotations}` and converted with the *same* converter as ARD-MAV:

```
py -3.13 -m src.data.prepare_ardmav --dataset ARD100 --split test --no-images
py -3.13 -m src.glad_detect --dataset ARD100 --pad released --out runs/exp005_glad_ard100
```

`--no-images` because `src.glad_detect` reads the `.mp4` and never opens an extracted
frame — 25 MB of labels instead of 30 GB of JPEGs. One caveat carries into any result:
ARD100 publishes **no scene-category grouping**, so the per-category rows EXP-004 reports
have no counterpart here; compare in aggregate and by `gt_size`. See
[prepare_ardmav.md](prepare_ardmav.md) and [glad_detect.md](glad_detect.md).

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

**Access.** `github.com/CZC-123/MOT-FLY` — **Apache-2.0**, commercially usable. Full
download recipe in [§ Download recipes](#mot-fly-recommended-first-addition) below.

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

> **Version confusion — now resolved.** Earlier notes flagged a 20-vs-14 sequence
> discrepancy as unverified. It is explained: **Dogfight re-annotated 14 of EPFL's 20
> sequences**, and its annotation folder holds exactly those 14, with non-contiguous ids —
> `Video_001, 011, 012, 018, 019, 029, 037, 046, 047, 048, 049, 053, 055, 056` — which is
> the signature of a subset selected from a larger pool. So the "14 videos / 38,948 frames"
> benchmark = EPFL imagery + Dogfight labels. Two lineages, not three: the **EPFL original**
> (20 sequences, ~8,000 manual boxes) and the **Dogfight re-annotation** (14 sequences).
> Published numbers from Dogfight, TransVisDrone and their successors all use the latter.

### How to download it — no Baidu, no permission needed

**A correction:** earlier notes said access was "via TransVisDrone prep". That is wrong.
TransVisDrone ships only *conversion scripts* and states the data "needs to be obtained
from permission with authors". The data and labels come from two other places, both open:

| Part | Source | Notes |
| --- | --- | --- |
| **Imagery** | EPFL CVLab — `https://drive.google.com/open?id=18CoTpjMs80dfanYNpbznjL4e-KB_Diel` | "Data and Code", linked from the [project page](https://www.epfl.ch/labs/cvlab/research/uav/research-unmanned-detection/). Covers both the UAV and aircraft sets. |
| **Labels** (14-seq benchmark) | Dogfight repo, plain git | `https://github.com/mwaseema/Drone-Detection/tree/main/annotations/FL-Drones-Dataset` |

```bash
# labels only -- ~0.5 MB, no auth
git clone --depth 1 https://github.com/mwaseema/Drone-Detection.git
ls Drone-Detection/annotations/FL-Drones-Dataset   # 14 files, Video_XXX.txt
```

**Annotation format** (verified) — one line per annotated frame, **frame numbers 0-based**,
boxes as **absolute `x1,y1,x2,y2` corners**, not xywh:

```
frame, n_boxes, x1, y1, x2, y2 [, x1, y1, x2, y2 ...]
0,1,348,339,411,366
```

Note this differs from ARD-MAV's VOC XML *and* from MOT-FLY's `x,y,w,h`, so it needs its
own converter. Frames with no line are frames with no annotation — the same
unannotated-vs-negative trap documented in
[MANIFEST.md](data/processed/ARD-MAV/MANIFEST.md); decide which before scoring.

If the EPFL Drive link fails, contact **Artem Rozantsev** (EPFL CVLab) via the project page.

**Verdict.** The best genuine held-out test for GLAD — GLAD published on ARD-MAV and
NPS-Drones but **not** FL-Drones. Now known to be obtainable without Baidu or a DUA. The
remaining friction is technical, not access: at 752×480 it is a different resolution from
GLAD's hardcoded 1080p motion constants, which must be rescaled first
([glad-model.md](glad-model.md) §6.3), and its largest targets fall outside the motion
branch's area cap regardless.

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

## MAV-VID — wrong size regime, but see the caveat

**Contents.** 64 videos / **40,232 images**: 53 videos (29,500) train, 11 videos (10,732)
val. **Single** multirotor target per frame. Originally from Rodriguez-Ramos et al., *IEEE
Access* 8:124451-124466 (2020); the numbers here are as the ICCVW-2021 benchmark defines it.

**How it was captured.** Quoting that benchmark:

> "It contains videos captured from **other drones, ground based surveillance cameras and
> handheld mobile devices**." … "UAVs usually move across the *x* axis and are **recorded
> from the bottom**."

So filter 3 holds for only part of it, the subset is not cleanly separable, and the dominant
framing — looking up at a target from below — is the ground-based signature, not a chase.

**The disqualifier.** Average object size is **136×77 px = 0.66% of image area**. Ours
averages **0.02%** (ARD-MAV) and **0.01%** (ARD100). That is a **~33–66× difference in
target area**. For scale, the same benchmark measures Drone-vs-Bird at 34×23 px (0.10%) —
so MAV-VID's targets are large even by ground-dataset standards. A detector tuned here
learns large-object features and tells us nothing about the 10–30 px regime that is the
entire problem.

### Is it worth a shot anyway? One specific case.

Not for detection benchmarking — the size gap makes any number from it non-transferable.

**But there is a genuine hole it fills.** Our ARD-MAV test split contains **zero targets
over 96 px** (`large` bucket: 0, see [MANIFEST.md](data/processed/ARD-MAV/MANIFEST.md)).
We have *no data at all* for the close-range regime. If the mission ends in an intercept,
a lock-on, or any terminal phase where the target grows to fill a meaningful part of the
frame, then the current data covers none of it — and MAV-VID is precisely that regime,
with multirotor targets and real video.

So: **not a training or benchmark set, but the only candidate for terminal-phase coverage.**
Worth revisiting if and when the requirement extends past detection-at-range. Until then it
is out of scope.

**Access.** Kaggle, "multirotor aerial vehicle vid mavvid". YOLO annotations; the benchmark
repo `github.com/KostadinovShalon/UAVDetectionTrackingBenchmark` ships
`convert_mav_vid_to_coco.py`. *(License unstated in the benchmark repo — unverified.)*

## UAVSwarm — resolved: **not air-to-air**. Fails filters 1 and 3.

Previously listed as "viewpoint unverified". **Now verified from the paper itself** (MDPI
*Remote Sensing* 14(11):2601). It does not qualify, on three independent counts.

**1. The viewpoint is explicitly the wrong two.** Quoting the dataset section:

> "The UAVSwarm dataset includes both **ground-to-air** UAV swarm and **air-to-ground** UAV
> swarm, which makes the background have complex and dynamic changes in the sky, ground,
> sky and ground, as well as background and light."

Ground-to-air is a camera on the ground looking up. Air-to-ground is a camera aloft looking
*down at the ground*. **Neither is air-to-air** — there is no chase geometry anywhere in it.

**2. Most cameras are static.** Tables 1 and 2 label every sequence `static` or `moving`,
and the clear majority are **static**. Ego-motion is the thing motion compensation exists
to cancel; a mostly-static set does not exercise it.

**3. The resolutions are sub-HD and ragged.** Per-sequence: 812×428, 446×270, 639×328,
847×412, 640×352, 863×467, 625×291, 720×479, 764×479, 844×455, 863×364, 810×475, 863×472 …
No two agree, none is HD, and the aspect ratios are inconsistent. That is the signature of
**footage cropped and rescaled from web video**, not a controlled capture — the same paper
criticises the "Real World" dataset for exactly this, noting its low resolution "because all
the data are obtained from YouTube videos, while other datasets are collected by researchers
themselves."

Its own **Data Availability Statement reads "Not applicable"**, so even the distribution
story is unclear.

**Contents, for the record.** 72 sequences, 12,598 images, 13 scenarios, 19+ UAV models,
3–23 UAVs per sequence, 30 fps, split 36/36 (6,844 / 5,754 images).

**Verdict. Do not use.** The airframe variety is real, but it is bought with the wrong
viewpoint, static cameras and sub-HD web footage. **Note that SCT-MOT groups it with
"air-to-air" benchmarks — that is wrong**, and is a good reminder to check the originating
paper rather than a citing one.

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

# How to check a dataset's viewpoint yourself

Papers advertise "UAV dataset" and "aerial" for four incompatible geometries, and **citing
papers get it wrong** — SCT-MOT calls UAVSwarm air-to-air; the originating paper says
ground-to-air and air-to-ground. Always go to the source. This is the order that resolves
it fastest:

1. **Read the originating paper's dataset section, not a citing one.** Search the PDF for
   `ground-to-air`, `air-to-air`, `air-to-ground`, `mounted on`, `static`, `handheld`. One
   sentence usually settles it. If the publisher blocks fetching (MDPI returns 403), the
   `mdpi-res.com/d_attachment/...` PDF mirror generally works, or use the arXiv version.
2. **Look for a per-sequence table.** The good datasets publish one, and it often has a
   literal **`Camera: static / moving`** column — that is what settled UAVSwarm. It also
   exposes resolution inconsistency.
3. **Check resolution consistency.** A controlled capture is one or two fixed resolutions
   (ARD-MAV: 1920×1080 throughout). A ragged spread of sub-HD sizes with mismatched aspect
   ratios means **footage scraped from web video** and rescaled.
4. **Check the target's aspect ratio in the label statistics.** ~1:1 to 1.5:1 is a
   multirotor; ~3:1 is a fixed-wing planform. This is how NPS-Drones was caught (65×21).
5. **Look at the background, not the target.** Air-to-air footage swings between sky and
   terrain within a single sequence and the horizon tilts with ego-motion. Ground-to-air is
   sky-dominated with a stable horizon. Air-to-ground has no sky at all.
6. **Last resort — measure it.** Pull ten frames and run the homography step of a motion
   compensator over consecutive pairs (`src/algo` has no such helper yet; GLAD's
   `motion_compensate` in [third_party/GLAD/Functions.py](third_party/GLAD/Functions.py) is
   a working reference). A static camera gives a near-identity homography; a flying platform
   does not. This is objective and takes minutes.

---

# Tier 3 — ground-based (auxiliary only)

Useful for airframe variety and bird negatives. **Never** for benchmarking this project.

| Dataset | Contents | Capture | Labels | License | Get it |
| --- | --- | --- | --- | --- | --- |
| **DUT Anti-UAV** | 10,000 images + 20 videos | ground/upward cameras, varied outdoor scenes, day & night | boxes, train/val/test provided | **Apache-2.0** | [github.com/wangdongdut/DUT-Anti-UAV](https://github.com/wangdongdut/DUT-Anti-UAV) — Google Drive + Baidu |
| **Anti-UAV410** | 410 **thermal IR** videos, 438k boxes, split 200/90/120 | ground thermal cameras "in the wild"; **>half of targets under 50 px** | boxes | research | [github.com/HwangBo94/Anti-UAV410](https://github.com/HwangBo94/Anti-UAV410) — Drive, Baidu code `a410` |
| **Anti-UAV** | 318 **paired RGB+IR** sequences, ~580k boxes, 25 fps MP4 | ground, registered dual-sensor; DJI + Parrot targets, day & night | boxes | research | [github.com/ZhaoJ9014/Anti-UAV](https://github.com/ZhaoJ9014/Anti-UAV) · [anti-uav.github.io](https://anti-uav.github.io) |
| **VisioDECT** | 20,924 images, **6 airframes** | ground, 30–100 m, 3 weather/lighting scenarios; 20 months, 12+ locations | boxes in `.txt`/`.xml`/`.csv` | IEEE DataPort | [ieee-dataport.org](https://ieee-dataport.org/) — search "VisioDECT" |
| **Drone-vs-Bird** | 77 videos, **95k+ frames** | ground static & pan-tilt, long range, **real birds in frame** | boxes incl. birds | **DUA, non-commercial** | [github.com/wosdetc/challenge](https://github.com/wosdetc/challenge) — signed DUA |
| **LRDDv2** | **39,516 still images** @1080p, range on 8k+ | DJI Mavic Air 2 + iPhone/Pixel; targets DJI Mini 3, Mavic Pro; to ~107 m | **YOLO boxes + range in metres** | see page | [research.coe.drexel.edu/ece/imaple/lrddv2](https://research.coe.drexel.edu/ece/imaple/lrddv2/) — **Google Forms request**, `kp3275@drexel.edu` |
| **LRDDv3** | adds **thermal** + range | ICRA 2026 | boxes + range | ICRA 2026 | [arXiv:2605.25942](https://arxiv.org/abs/2605.25942) |
| **MMAUD** | ~15 sequences, **rosbags**, 11.1–19.7 GB each | stereo + Livox Avia + Mid-360 + mmWave + 4 mic arrays; targets Mavic 2/3, Phantom 4, Avata, M300 | **Leica 3D ground truth**, type + trajectory | **CC BY-NC-SA** | [github.com/ntu-aris/MMAUD](https://github.com/ntu-aris/MMAUD) — OneDrive + Drive |
| **MM-UAV** | 1,321 seq, ~2.8M frames/modality | ground **RGB + IR + event camera**; RGB only 640×360 | boxes + persistent IDs | CC BY-NC-SA | [xuefeng-zhu5.github.io/MM-UAV](https://xuefeng-zhu5.github.io/MM-UAV/) — pending release |
| **UETT4K** | 4K images | ground, diverse conditions | boxes | IEEE | [IEEE Xplore 10971965](https://ieeexplore.ieee.org/document/10971965/) |
| **USC-Drone** | video | **handheld, from the ground**; single viewpoint, many frames unlabelled | boxes (incomplete) | research | noted for exclusion |
| **"Real World" (RWOQ)** | 56,821 images / 55,539 boxes | **scraped from YouTube**, hand-labelled; low res, flat/elevation views | boxes | research | widest airframe variety, worst image quality |

### Which of these are actually *video* of a ground camera tracking a multirotor

Six of the ten are video; the rest are stills or a non-video format. Ranked by usefulness
here:

| Dataset | Sequence volume | Modality | Friction |
| --- | --- | --- | --- |
| **Anti-UAV300** ⭐ | ~300 sequence pairs, Full HD | **RGB + IR** | **Google Drive, open** — best RGB volume |
| **DUT tracking** ⭐ | 20 sequences (frames + GT archives) | RGB | **Apache-2.0**, Google Drive — easiest |
| **Drone-vs-Bird** | 77 sequences, 95k+ frames | RGB | **signed DUA, non-commercial** |
| **Anti-UAV410** | 410 sequences, 438k boxes | **IR only** | Drive/Baidu |
| **Anti-UAV600** | — | **IR only** | ModelScope (Chinese host) |
| **MM-UAV** | 1,321 sequences | RGB 640×360 + IR + event | **not yet released** |
| **USC-Drone** | handheld ground | RGB | single viewpoint, many frames unlabelled |
| DUT *detection* · VisioDECT · LRDDv2/v3 · UETT4K | — | **sparse stills** | unusable for motion |
| MMAUD | — | **rosbags** | different pipeline entirely |

**Anti-UAV300 is the answer** to "ground camera tracking a multirotor, in sequence form,
downloadable now": Full HD, RGB **and** IR, DJI and Parrot targets, day and night, varied
backgrounds — and it has a plain Google Drive link. Note the family splits by modality:
**300 = RGB+IR, 410 = IR only, 600 = IR only.** Only the 300 gives you RGB. Its known
weakness is that the RGB and IR streams are **not aligned in time or space**, which matters
only if you wanted the fusion.

**DUT's tracking half** is the fastest thing to get working today — Apache-2.0, Google
Drive, no DUA, no Baidu — but it is only 20 sequences, and you must take the **tracking**
archives, not the 10,000-image detection subset.

*(The DJI-and-Parrot / day-and-night details come from the UAVSwarm paper's related-work
section rather than the Anti-UAV paper itself — unverified at source.)*

Remember what this data can and cannot do: a **static ground camera** makes frame
differencing trivial and removes the ego-motion the whole motion-compensation stage exists
to cancel. Use these for airframe appearance and bird discrimination, never to measure
detection performance for this project.

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

Repo: `github.com/CZC-123/MOT-FLY` · **Apache-2.0** · contact `3120210041@bit.edu.cn`

**Two mirrors — Google Drive needs no account:**

| Mirror | URL | Note |
| --- | --- | --- |
| **Google Drive** | `https://drive.google.com/file/d/1GiWLF8B18FGDcCSuSuvGokczCkP_NEgo/view?usp=sharing` | **use this one** |
| Baidu | `https://pan.baidu.com/s/1eS84Ooz0URojz1tAJNZ5Eg?pwd=pe53` | password `pe53` |

It is a single Drive file. Browser download is simplest; for a headless GPU instance use
`gdown`, which handles Drive's large-file confirmation page:

```bash
py -3.13 -m pip install gdown
gdown 1GiWLF8B18FGDcCSuSuvGokczCkP_NEgo -O MOT-FLY.zip
unzip MOT-FLY.zip -d data/raw/
```

**Layout** — MOTChallenge, already split into `train/` and `test/`:

```
MOT-FLY/
├── train/
│   ├── DJI_0003_D_S_E/          # sequence names carry the DJI source clip id
│   │   ├── img1/000001.jpg …    # frames, 1-based 6-digit, 1920×1080
│   │   ├── gt/gt.txt            # frame, id, x, y, w, h, conf, class, visibility
│   │   ├── det/det.txt          # supplied detections (ignore; we run our own)
│   │   └── seqinfo.ini          # frame count, resolution, image extension
│   └── … 8 sequences total
└── test/  … 8 sequences
```

### ⚠️ The Drive link is dead — treat MOT-FLY as unobtainable for now

**Status 2026-08-17: the Google Drive file returns "file does not exist"** — deleted or
moved, not a quota block. My earlier quota diagnosis was wrong.

This is not a transient failure. Repository **issue #1, titled 「链接失效」 ("link
invalid"), was opened 2024-12-27** asking the author to re-share, and **has received no
reply in ~20 months.** The repo has no other activity, and no mirror exists anywhere I
could find.

Remaining routes, in order:

1. **Try the Baidu link** — `https://pan.baidu.com/s/1eS84Ooz0URojz1tAJNZ5Eg?pwd=pe53`
   (password `pe53`). It is a separate host and may still be live; the issue does not say
   which link broke. Subject to the Baidu access problems below.
2. **Email the author** — `3120210041@bit.edu.cn` (Beijing Institute of Technology). Cite
   the paper (*"An Experimental Evaluation Based on New Air-to-Air Multi-UAV Tracking
   Dataset"*) and mention issue #1. Given the silence on GitHub, set expectations low.

**Consequence: MOT-FLY cannot be planned around.** It was listed as the best free addition
and as the easy M4b option — **both claims are withdrawn** until a link is confirmed
working. M4b falls back to **FL-Drones** (primary) and **ARD100-extra** (secondary).

**Converting to our canonical layout.** `gt.txt` is `frame,id,x,y,w,h,…` in **absolute
top-left pixel** coords, whereas our YOLO labels are normalised centre-xywh — the same
conversion `src.data.prepare_ardmav` already does for VOC, just from a different source.
Two things to decide up front:

- **The `id` column.** Dropping it loses the track association. Keep it in a sidecar
  (`tracks.jsonl`) — MOT-FLY is the only multi-target set we would hold, so it is the only
  data that could ever support track-level evaluation.
- **Genuine negatives.** Unlike ARD-MAV, absence of a `gt.txt` row for a frame means the
  target is absent, not unannotated — `seqinfo.ini` gives the authoritative frame count, so
  frames with no row are real negatives and should be emitted as empty labels.

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

## ARD100, and Baidu-gated data generally, from outside China

> ## ✅ Resolved 2026-08-18 — ARD100 is on Zenodo, no Baidu needed
>
> A community mirror exists and is **open access**: **DOI
> [10.5281/zenodo.15870538](https://doi.org/10.5281/zenodo.15870538)** — a single
> `ARD100.zip`, **27.35 GB**, **CC-BY-4.0**, uploaded by Yu-Hsi Chen (`@wish44165`) and
> **endorsed by the YOLOMG author**, who wrote in issue #7 "You can send the download link
> to me, and I will paste it in the readme."
>
> ```bash
> # --read-timeout is essential: without a stall guard the transfer hangs
> # forever on a half-open socket (cost 2.6 h at 45% when the machine idled).
> wget -c --tries=0 --waitretry=15 --read-timeout=60 \n>   "https://zenodo.org/records/15870538/files/ARD100.zip?download=1" -O ARD100.zip
> ```
>
> Verified directly: `HTTP 200`, `content-length: 27351535415`, and range requests return
> `206 Partial Content`, so **`-c` resume works**. Zenodo intermittently returns `504`
> under load — one in two probes during testing — so resume is not optional on a 27 GB
> file. Use `wget -c` (or `curl -C -`) and re-run until it completes; add
> `--tries=0 --waitretry=15` to make it self-healing.
>
> Pull it **onto the GPU instance** for M7, not here. For M4b, 10–15 videos suffice — but
> the archive is a single zip, so selective extraction happens after download
> (`unzip -l` to list, then extract only the wanted members).
>
> **How it was found, because the method generalises:** the dataset repo's *issue tracker*.
> `Irisky123/YOLOMG` issue #7 ("How to download ARD100 dataset") runs to 64 comments of
> international users hitting exactly this wall, and the mirror surfaced there long before
> any README mentioned it. Check issues before concluding a dataset is unobtainable — the
> same sweep also confirmed FL-Drones' real home (issue #3, answered by the author with the
> EPFL CVLab link) and that MOT-FLY's link has been dead since 2024 with no reply.

The original Baidu link still exists (`https://pan.baidu.com/s/1ycAoKbzQ1rlzvKr8VRakgw?pwd=1x2z`,
code `1x2z`) and reportedly still works, but the notes below explain why it is the worse
route.

**Baidu is not geo-blocked from Israel.** The obstacles are different, and the second one
is the real problem:

| Obstacle | Reality |
| --- | --- |
| **Registration** | Historically required a mainland `+86` number. The usual workaround is registering with a **Chinese email** (`@qq.com`, `@163.com`) as the login instead of a phone. |
| **Throttling** | **This is the blocker.** Without a paid SVIP account, Baidu throttles browser downloads to roughly 100 KB/s. ARD100 is 100 videos of 1080p footage — tens of GB. At that rate the download runs for **days to weeks**, and resuming is unreliable. |

So even a successful registration does not make ARD100 practically obtainable this way.

**On third-party "Baidu downloader" sites:** not recommended. They are frequently paywalled
or outright malicious, you hand a stranger the share link, and none of them solve the size
problem. Some also breach Baidu's terms. If you try one, treat it as untrusted.

### The route that actually works: ask the authors

**ARD100's lead author is Hanqing Guo — the same lead author as GLAD, same lab** (Westlake
University, senior author Shiyu Zhao). Two things make this a strong ask:

- That group **already publishes a Google Drive mirror for ARD-MAV**, so they are plainly
  willing to host outside Baidu.
- The GLAD README carries an explicit invitation: *"If you have any problem when using this
  dataset, please feel free to contact: **guohanqing@westlake.edu.cn**."*

**Ask for the subset, not the whole thing.** M4b does not need all 202,467 frames — it needs
only the ARD100 videos that are **not** among our local ARD-MAV 60. That is a much smaller
transfer, a far more answerable request, and precisely what the experiment calls for. Say
which paper you are citing, that Baidu is not usable from your location, and offer to accept
a Drive/OneDrive link or a subset.

**Same pattern for the other Baidu-gated sets:** Det-Fly lists a OneDrive mirror alongside
Baidu (`zhengye@westlake.edu.cn` — same institution), and Anti-UAV410 and DUT Anti-UAV both
already offer Google Drive. **ARD100 is the only one in this survey with no non-Baidu
option**, which is exactly why the email is worth sending.

---

# Traps

- **VisDrone is not a drone-detection dataset.** Cars and pedestrians shot *from* a drone.
  Same for MAVREC, AU-AIR, MOR-UAV, SynDrone — all look *down at the ground*.
  **Air-to-air ≠ aerial.**
- **Check the target airframe.** NPS-Drones (delta-wing) and MMFW-UAV (fixed-wing) are
  air-to-air but not multirotor. AOT is mostly manned aircraft.
- **Check the annotation version.** NPS-Drones ships v1 and v2 from Purdue *and* has a third
  re-annotation from Dogfight, whose 50 files are on GitHub. FL-Drones has two lineages:
  EPFL's original 20 sequences and Dogfight's re-annotated 14. Published
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
