"""Why do static things light up? Residual misalignment and texture inside blobs vs elsewhere."""
import json
import cv2, numpy as np
from src.algo.glad import vendor
from src.algo.glad.motion import UPSTREAM
from src.data.hud_mask import load_mask

FR = {750, 850, 925, 958, 962}
sess = json.load(open("data/processed/SOFA-O4/annotations/first_catch.json"))["frames"]
boxes = {int(k): v["box"] for k, v in sess.items() if v and v.get("box")}
hud = load_mask("data/processed/SOFA-O4/hud_mask.png")
port = vendor.import_motion_port(lambda _: 0, UPSTREAM, hud_mask=hud)
cap = cv2.VideoCapture("data/processed/SOFA-O4/videos/first_catch.avi")
prev, n = None, 0
agg = {k: [] for k in ["res_in", "res_out", "grad_in", "grad_out", "prod_in", "prod_out", "thr"]}
while True:
    ok, fr = cap.read()
    if not ok: break
    n += 1
    if n in FR:
        p, c = port._prepare(prev), port._prepare(fr)
        comp, border, _ = port.compensate(p, c)
        diff = cv2.absdiff(c, comp)
        thr = 5 + int(diff.mean())
        binary = port._binary(diff, thr, border, median=True)
        x, y, w, h = [int(v) for v in boxes[n]]
        fg = binary > 0
        fg[max(y - h, 0):y + 2 * h, max(x - w, 0):x + 2 * w] = False   # drop the drone
        valid = (border == 0) & ~hud
        valid[max(y - h, 0):y + 2 * h, max(x - w, 0):x + 2 * w] = False
        # remaining misalignment after GLAD's warp: dense flow compensated-prev -> current
        flow = cv2.calcOpticalFlowFarneback(comp, c, None, 0.5, 4, 21, 5, 7, 1.5, 0)
        res = np.linalg.norm(flow, axis=2)
        gx, gy = cv2.Sobel(c, cv2.CV_32F, 1, 0, ksize=3) / 8, cv2.Sobel(c, cv2.CV_32F, 0, 1, ksize=3) / 8
        grad = np.hypot(gx, gy)  # grey levels per pixel
        inn, out = fg & valid, ~fg & valid
        print(f"frame {n}: thr {thr} | residual px  in-blob med {np.median(res[inn]):.2f}  outside {np.median(res[out]):.2f}"
              f" | gradient gl/px in {np.median(grad[inn]):.1f}  out {np.median(grad[out]):.1f}"
              f" | grad x residual in {np.median((grad*res)[inn]):.1f}  out {np.median((grad*res)[out]):.1f}")
        # the HUD: screen-fixed pixels the warp moves
        # save a residual-flow heatmap for the user
        vis = cv2.applyColorMap(np.clip(res * 60, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        vis[~valid] = (vis[~valid] * 0.3).astype(np.uint8)
        cv2.putText(vis, f"frame {n}: motion left after GLAD's warp (bright = 4+ px)", (10, 40), 0, 1, (255, 255, 255), 2)
        cv2.imwrite(f"runs/sofa_o4/exp012b_motion_check/residual_{n}.jpg", np.hstack([fr, vis]))
    prev = fr
