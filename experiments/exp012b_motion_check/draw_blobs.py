"""Draw MOD2_global's blobs onto first_catch: stills for chosen frames, and a video of 708-964."""
import json, sys
import cv2, numpy as np
from src.algo.glad import vendor
from src.algo.glad.motion import UPSTREAM
from src.data.hud_mask import load_mask

OUT = "runs/sofa_o4/exp012b_motion_check/"
STILLS = {750, 850, 925, 958, 962}
sess = json.load(open("data/processed/SOFA-O4/annotations/first_catch.json"))["frames"]
boxes = {int(k): v["box"] for k, v in sess.items() if v and v.get("box")}
hud = load_mask("data/processed/SOFA-O4/hud_mask.png")
port = vendor.import_motion_port(lambda _: 0, UPSTREAM, hud_mask=hud)
cfg = UPSTREAM
GREY, ORANGE, RED, GREEN, WHITE = (150, 150, 150), (0, 165, 255), (0, 0, 255), (0, 255, 0), (255, 255, 255)

def legend(img, n, stats):
    cv2.rectangle(img, (0, 0), (img.shape[1], 92), (0, 0, 0), -1)
    cv2.putText(img, f"frame {n}   " + stats, (10, 28), 0, 0.8, WHITE, 2)
    x = 10
    for col, txt in [(GREY, "rejected (size/shape)"), (ORANGE, "candidate"),
                     (RED, "candidate, top-10 by area (#rank)"), (GREEN, "blob on the drone"),
                     (WHITE, "your label")]:
        cv2.rectangle(img, (x, 50), (x + 26, 76), col, -1)
        cv2.putText(img, txt, (x + 34, 72), 0, 0.55, WHITE, 1)
        x += 34 + 10 * len(txt) + 20

def overlaps(b, t):
    x, y, w, h = b; tx, ty, tw, th = t
    ix = max(0, min(x + w, tx + tw) - max(x, tx)); iy = max(0, min(y + h, ty + th) - max(y, ty))
    return ix * iy > 0.1 * min(w * h, tw * th)

cap = cv2.VideoCapture("data/processed/SOFA-O4/videos/first_catch.avi")
vid, prev, n = None, None, 0
while True:
    ok, fr = cap.read()
    if not ok: break
    n += 1
    if prev is not None and n in boxes:
        p, c = port._prepare(prev), port._prepare(fr)
        comp, border, _ = port.compensate(p, c)
        diff = cv2.absdiff(c, comp)
        binary = port._binary(diff, cfg.global_threshold_base + int(diff.mean()), border, median=True)
        t = [int(round(v)) for v in boxes[n]]
        blobs = []
        for ct in cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]:
            x, y, w, h = cv2.boundingRect(ct)
            a = cv2.contourArea(ct)
            if h == 0: continue
            ok_ = cfg.blob_area[0] < a < cfg.blob_area[1] and cfg.global_blob_ratio[0] < w / h < cfg.global_blob_ratio[1]
            blobs.append((ct, (x, y, w, h), a, ok_, overlaps((x, y, w, h), t)))
        cands = sorted([b for b in blobs if b[3]], key=lambda b: -b[2])
        rank = {id(b[0]): i + 1 for i, b in enumerate(cands)}
        img = fr.copy()
        for ct, bb, a, ok_, on in blobs:
            if not ok_ and not on:
                cv2.drawContours(img, [ct], -1, GREY, 1)
        for ct, (x, y, w, h), a, ok_, on in blobs:
            if ok_ and not on:
                r = rank[id(ct)]
                col = RED if r <= 10 else ORANGE
                cv2.drawContours(img, [ct], -1, col, 2)
                if r <= 10:
                    cv2.putText(img, f"#{r}", (x, max(y - 4, 100)), 0, 0.6, RED, 2)
        cv2.rectangle(img, (t[0], t[1]), (t[0] + t[2], t[1] + t[3]), WHITE, 2)
        drone = [b for b in blobs if b[4]]
        for ct, (x, y, w, h), a, ok_, on in drone:
            cv2.drawContours(img, [ct], -1, GREEN, 3)
        dtxt = ("drone blob: " + ", ".join(
            f"{int(a)}px2 {'#'+str(rank[id(ct)]) if ok_ else 'REJECTED'}" for ct, _, a, ok_, _ in drone)
            if drone else "no blob on the drone")
        legend(img, n, f"{len(cands)} candidates, {len(blobs)} blobs   |   {dtxt}")
        if vid is None:
            vid = cv2.VideoWriter(OUT + "blobs_708_964.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 10, img.shape[1::-1])
        vid.write(img)
        if n in STILLS:
            cv2.imwrite(OUT + f"blobs_{n}.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            # zoom: the drone and the ten biggest candidates, as 3x crops of frame + binary
            tiles = []
            for label, (x, y, w, h), col in ([("drone", tuple(t), GREEN)] +
                                             [(f"#{rank[id(b[0])]}", b[1], RED) for b in cands[:10] if not b[4]][:7]):
                cx, cy, s = x + w // 2, y + h // 2, max(w, h, 40) + 30
                x0, y0 = max(cx - s // 2, 0), max(cy - s // 2, 0)
                crop = img[y0:y0 + s, x0:x0 + s]; bcrop = binary[y0:y0 + s, x0:x0 + s]
                raw = fr[y0:y0 + s, x0:x0 + s]
                trio = [cv2.resize(z if z.ndim == 3 else cv2.cvtColor(z, cv2.COLOR_GRAY2BGR), (240, 240), interpolation=cv2.INTER_NEAREST) for z in (raw, crop, bcrop)]
                tile = np.vstack(trio)
                cv2.putText(tile, label, (6, 26), 0, 0.8, col, 2)
                tiles.append(cv2.copyMakeBorder(tile, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=col))
            cv2.imwrite(OUT + f"blobs_{n}_zoom.jpg", np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 92])
    prev = fr
vid.release()
