# Hardware constraints

Measured on this machine, 2026-08-13.

| | |
| --- | --- |
| CPU | 12th Gen Intel Core i7-1255U (15 W mobile, 2P + 8E) |
| RAM | 16 GB |
| GPU | Intel Iris Xe (integrated) — **no NVIDIA, no CUDA** |
| Disk free | ~210 GB on `C:` |
| Python | 3.13.11 (`py -3.13`) — 3.14 is also installed but runs ahead of the torch/ultralytics wheel matrix |
| torch | 2.9.1**+cpu** — `torch.cuda.is_available()` → `False`, `torch.xpu.is_available()` → `False` |
| ultralytics | 8.3.235 |

## What this means for the plan

**Inference is fine locally.** Running a pretrained YOLOv8s over a few thousand frames
at 640–1280 px is minutes, not hours. Step 1 of the plan (the baseline sanity check)
runs here today.

**Training is not.** Fine-tuning GLAD or YOLOMG on ARD100 (202k frames) on a 15 W
CPU is a multi-week job. It needs a rented GPU:

| Option | Notes |
| --- | --- |
| **Kaggle Notebooks** | 2× T4, 30 h/week free. Best free tier. 20 GB working disk is the constraint — stage a subset. |
| **Google Colab** | Free T4 with session limits; Pro ~$10/mo for longer runs and better GPUs. |
| **RunPod / Vast.ai** | ~$0.20–0.35/hr for a 3090/4090. Best value once runs get long. |
| **Lambda** | Cleaner UX, higher price. |

Plan the data staging around this: don't download all of ARD100 + Det-Fly locally and
then try to push it to a cloud box. Det-Fly alone is 13,271 images at 3840×2160
(~50 GB+). Pull datasets **directly onto the GPU instance**, and keep only a small
local subset for pipeline debugging.

## Speeding up local inference

The Iris Xe is unused by the CPU torch build. Two ways to exploit it, if local
inference speed ever matters (e.g. for a real-time demo):

- **OpenVINO** — ultralytics exports to it natively
  (`model.export(format="openvino")`), and it targets the Intel iGPU. Typically a solid
  speedup over CPU torch for inference. `pip install openvino`.
- **torch XPU build** — install the Intel-flavoured torch wheel. More invasive; would
  replace the current `+cpu` install. Not worth it unless OpenVINO falls short.

Neither helps with training.
