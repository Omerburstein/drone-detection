"""A CPU-runnable port of GLAD's released detection pipeline.

GLAD (Guo et al., IEEE T-ITS 2024) is the reference point for this project: it
released the ARD-MAV dataset, our test split is its official split verbatim, and
its per-condition figures are the only published numbers our harness can be
checked against. See docs/glad-model.md.

The released repository is vendored at `third_party/GLAD/`. Its entry point
cannot run here — it loads TensorRT engines onto a hardcoded second CUDA device
— so this package supplies the parts that have to change and reuses everything
else unmodified:

| Upstream | Here | Why |
| --- | --- | --- |
| `detector{1,2,3}_trt.py` | `yolo.py` | TensorRT 7.2 + `cuda.Device(1)`; the `.pt` checkpoints ship alongside the engines and load on CPU |
| `Functions.Mynet_infer` | `classifier.py` | upstream re-reads the checkpoint from disk on every candidate box of every frame |
| `MOD2.py` motion modules | used verbatim via `vendor.py` | classical CV, no GPU, nothing to port |
| `GLAD.py` display loop | `pipeline.py` | a module-level `while` with `imshow` is not callable |

Everything with a number attached to it — thresholds, region sizes, selection
rules, state transitions — is reproduced exactly, including the quirks. This is
a fidelity exercise: the point of M4a is that a *large* gap against the paper
means our harness has a bug, so any silent "improvement" here would destroy the
measurement.
"""
