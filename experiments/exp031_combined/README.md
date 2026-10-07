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

Seven overlays per clip and radius, on catch_2, catch_4 and catch_5, from EXP-030's dumps
(no detector re-run):

- `exp028` — max score, gate. At merge 20 this is EXP-028's default.
- `exp029` — `exp028` plus `--min-move 20`. At merge 20 on catch_2 this is EXP-029's stacked run.
- `all` — `exp029` plus `--merge-score sum`. Everything, at c-keep 6.
- `all_ck{8,10,12,15}` — `all` with a stricter `--c-keep`. This is the matched-load sweep
  EXP-030 left open: under the sum, clusters of weak blobs clear c-keep 6 and raise the load.

Outputs go to `runs/sofa_analog/exp031_combined/merge<R>/<clip>/{variant}_<clip>_<start>_<end>`:
`.mp4`, `.csv`, `_drone.csv`, `_frames.csv` and a `.log`. The batch log and the table are
in `runs/sofa_analog/exp031_combined/run_all.log`.

## Result

See EXP-031 in `docs/experiments.md`.
