# EXP-030 — merged objects scored by the sum of their members' c

The user's question: when `--merge 20` folds blobs into one object, does the object take
the strongest member's `c` or the sum? It took the max: the anchor's `c` alone. This
experiment scores it by the **sum** instead.

`overlay_split.py --merge-score sum` (in `exp025_top3/`). The default is `max`, which keeps
the old behaviour. With `sum`:

- anchors are still chosen by their own `c`, strongest first, as before;
- the object's `c` is the sum of every member's `c`, and the anchor's own is kept as
  `c_max` in the shown CSV;
- the sections' tests are unchanged: sky passes if any member has c >= 6, ground if a
  window survivor is within 9 px of any member;
- everything downstream reads the summed `c`: the pooled ranking, the kinematic gate's
  `--c-keep` test for weak continuation, and its track evidence.

Every blob in the bubble counts, including the sub-threshold ones (most sit at c 3–4).

## Running

```bash
py -3.13 experiments/exp030_merge_sum/run_clips.py --merge 20 30   # resumable; --jobs 4
```

Four overlays per clip, on catch_2, catch_4 and catch_5, all at EXP-028's settings: top 3,
circles >= 20 px, `--osd-grid`. Each merge radius given to `--merge` (default 20) gets its
own subfolder; merge 20 and merge 30 have been run.

- `max_nogate` and `sum_nogate` are the pooled ranking alone, as EXP-027 runs it.
- `max_gate` and `sum_gate` go through EXP-028's default kinematic gate (c-keep 6, ranked
  by track evidence). **`sum_gate` is the overlay.**

At merge 20 the `max_` runs re-create EXP-028's `nogate` and `gate`. All 18 of their CSVs
match EXP-028's byte for byte, so `--merge-score` changes nothing when it is off. At merge
30 the `max_` runs are that radius's own baseline.

Outputs go to
`runs/sofa_analog/exp030_merge_sum/merge<R>/<clip>/{variant}_<clip>_<start>_<end>`:
`.mp4`, `.csv` (every shown object, with `c_max` on the sum runs), `_drone.csv`,
`_frames.csv` and a log. The batch logs are `exp030_merge_sum.log` (merge 20) and
`exp030_merge_sum/merge30.log`.

## Result

See EXP-030 in `docs/experiments.md` for the table and the reading.
