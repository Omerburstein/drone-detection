# EXP-031 — every stage at once

The user asked for the past experiments combined into one. Each of EXP-025d to EXP-030
added one flag to `overlay_split.py` (in `exp025_top3/`). They were measured one on top of
the previous, but never all on together, and the EXP-029 stack was only ever run on catch_2:

| stage | flag | from |
| --- | --- | --- |
| sky branch on stage-1 sky, 2-of-4 window on the ground, pooled top 3 | `--top 3` | EXP-025d |
| blobs within R px folded into the strongest | `--merge 20` | EXP-025d |
| OSD horizon dashes dropped by the character grid | `--osd-grid` | EXP-027 |
| speed limit + hysteresis track, ranked by evidence | `--kinematic` | EXP-028 |
| moving factor, the speed limit's bound from below | `--min-move 20` | EXP-029 |
| merged object scored by the sum of its members' c | `--merge-score sum` | EXP-030 |

## Running

```bash
py -3.13 experiments/exp031_combined/run_all.py --merge 20 30   # resumable; --jobs 4
```

Twelve overlays per clip and radius, on catch_2, catch_4 and catch_5, from EXP-030's dumps
(no detector re-run):

- `exp028` — max score, gate. At merge 20 this is EXP-028's default.
- `exp029` — `exp028` plus `--min-move 20`. At merge 20 on catch_2 this is EXP-029's stacked run.
- `exp030` — `exp028` plus `--merge-score sum`, without the moving factor. This is EXP-030's
  `sum_gate`, and it must match that run byte for byte at both radii.
- `exp030_ck{8,10,12,15}` — `exp030` with a stricter `--c-keep`. This is the matched-load
  sweep EXP-030 left open: under the sum, clusters of weak blobs clear c-keep 6 and raise
  the load.
- `all` — `exp029` plus `--merge-score sum`. Everything, at c-keep 6.
- `all_ck{8,10,12,15}` — `all` with the same sweep, so the two sweeps differ by the moving
  factor alone.

Outputs go to `runs/sofa_analog/exp031_combined/merge<R>/<clip>/{variant}_<clip>_<start>_<end>`:
`.mp4`, `.csv`, `_drone.csv`, `_frames.csv` and a `.log`. The batch log and the table are
in `runs/sofa_analog/exp031_combined/run_all.log`.

## The FIELD `.raw`, at the sensor's resolution

```bash
py -3.13 experiments/exp031_combined/run_field_raw.py            # 4128 wide; --width, --jobs 5
```

`all` on the FIELD capture's `.raw`: 4128x3008 Bayer, the same 3600 frames as the mp4 at 4x
the resolution (`src/data/raw_bayer.py`). There are no EXP-030 dumps for it, so the driver
builds the sky-branch and window dumps first. It runs over the three episodes in
`data/raw/FIELD/PROVENANCE.md`, 2–180, 1090–1553 and 3130–3600, at merge 20 and 30.
`field_raw/clipcfg.py` is the clip config. It routes the `.raw` path in `cv2.VideoCapture`
to `RawBayerCapture`, and `FIELD_RAW_WIDTH` sets the working width. Outputs go to
`runs/field/exp031_combined/raw<W>/<a>_<b>/`: `sky_candidates.csv`, `window_seeds_k4.csv`,
and `all_merge<R>_<a>_<b>` with `.mp4`, `.csv`, `_frames.csv` and `.log`. **Watch
`_1080p.mp4`:** the full-size `.mp4` is `mp4v` at 4128x3008, which Windows players will not
open. The driver writes an H.264 copy at 1080 rows through Media Foundation.

## Result

See EXP-031 in `docs/experiments.md`. In short, pooled over the three clips:
`--merge 30 --merge-score sum --c-keep 10` (the `exp030_ck10` row at merge 30) beats EXP-028
on every column at 13% less load, and catch_5 is the one clip that loses #1 frames. The
moving factor costs catch_5 13 drone frames and adds nothing once c-keep is raised.
