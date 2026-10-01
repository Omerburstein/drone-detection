"""Every candidate blob MOD2_global sees on first_catch: size, distance to the drone, strength."""
import csv, json, sys
import cv2, numpy as np
from src.algo.glad import vendor
from src.algo.glad.motion import UPSTREAM, candidate_score
from src.data.hud_mask import load_mask

sess = json.load(open("data/processed/SOFA-O4/annotations/first_catch.json"))["frames"]
boxes = {int(k): v["box"] for k, v in sess.items() if v and v.get("box")}
hud = load_mask("data/processed/SOFA-O4/hud_mask.png")
port = vendor.import_motion_port(lambda _: 0, UPSTREAM, hud_mask=hud)
cfg = UPSTREAM
out = open(sys.argv[1], "w", newline="")
wr = csv.writer(out)
wr.writerow(["frame", "drone_size", "area", "w", "h", "ratio", "dist", "on_drone", "strength", "shape_score", "passes"])
cap = cv2.VideoCapture("data/processed/SOFA-O4/videos/first_catch.avi")
prev, n = None, 0
while True:
    ok, fr = cap.read()
    if not ok: break
    n += 1
    if prev is not None and n in boxes:
        p, c = port._prepare(prev), port._prepare(fr)
        comp, border, _ = port.compensate(p, c)
        diff = cv2.absdiff(c, comp)
        binary = port._binary(diff, cfg.global_threshold_base + int(diff.mean()), border, median=True)
        tx, ty, tw, th = boxes[n]; tc = np.array([tx + tw / 2, ty + th / 2])
        for ct in cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]:
            x, y, w, h = cv2.boundingRect(ct)
            area = cv2.contourArea(ct)
            if h == 0: continue
            m = np.zeros(binary.shape, np.uint8); cv2.drawContours(m, [ct], -1, 1, -1)
            strength = float(diff[m > 0].mean()) if m.any() else 0.0
            ix = max(0, min(x + w, tx + tw) - max(x, tx)); iy = max(0, min(y + h, ty + th) - max(y, ty))
            on = ix * iy > 0.1 * min(w * h, tw * th)
            passes = cfg.blob_area[0] < area < cfg.blob_area[1] and cfg.global_blob_ratio[0] < w / h < cfg.global_blob_ratio[1]
            wr.writerow([n, max(tw, th), area, w, h, round(w / h, 3),
                         round(float(np.linalg.norm(np.array([x + w / 2, y + h / 2]) - tc)), 1),
                         int(on), round(strength, 2), round(candidate_score(area, w, h), 3), int(passes)])
    prev = fr
