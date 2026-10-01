"""Where in MOD2_global does the labelled drone get lost? Frame by frame."""
import json, sys, collections
import cv2, numpy as np
from src.algo.glad import vendor
from src.algo.glad.classifier import load_gate
from src.algo.glad.motion import UPSTREAM, CLUTTER, find_candidates, select, _moving_coherently
from src.data.hud_mask import load_mask

sess = json.load(open("data/processed/SOFA-O4/annotations/first_catch.json"))["frames"]
boxes = {int(k): v["box"] for k, v in sess.items() if v and v.get("box")}
hud = load_mask("data/processed/SOFA-O4/hud_mask.png")
gate = load_gate()
import Functions

def iou_hit(c, t):  # candidate box overlaps / sits within the target box
    x, y, w, h = c; tx, ty, tw, th = t
    ix = max(0, min(x + w, tx + tw) - max(x, tx)); iy = max(0, min(y + h, ty + th) - max(y, ty))
    return ix * iy > 0.1 * min(w * h, tw * th)

def diagnose(port, cfg, f1, f2, t):
    prev, cur = port._prepare(f1), port._prepare(f2)
    comp, border, _ = port.compensate(prev, cur)
    diff = cv2.absdiff(cur, comp)
    thr = cfg.global_threshold_base + int(np.mean(diff))
    binary = port._binary(diff, thr, border, median=True)
    contours, _ = cv2.findContours(binary.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    on_t = [c for c in contours if iou_hit(cv2.boundingRect(c), t)]
    if not on_t:
        return "no blob on target", {}
    areas = [cv2.contourArea(c) for c in on_t]
    cands = find_candidates(binary, cfg.blob_area, cfg.global_blob_ratio)
    info = dict(n=len(cands), areas=[int(a) for a in areas])
    tc = [b for b, _ in cands if iou_hit(b, t)]
    if not tc:
        big = max(areas)
        return ("blob too big" if big >= cfg.blob_area[1] else "blob fails area/ratio"), info
    kept = select(cands, cfg.max_candidates_global, cfg.rank_when_crowded)
    if not any(iou_hit(b, t) for b in kept):
        return ("bail: >50 candidates" if not cfg.rank_when_crowded else "ranked out"), info
    H, W = f1.shape[:2]
    for x0, y0, w0, h0 in kept:
        x1, y1, w1, h1 = port.enlargebox(x0, y0, w0, h0, cfg.enlarge, W, H)
        mine = iou_hit((x0, y0, w0, h0), t)
        if not cfg.global_blob_ratio[0] < w1 / h1 < cfg.global_blob_ratio[1]:
            if mine: return "enlarged ratio", info
            continue
        if not _moving_coherently(comp[y1:y1+h1, x1:x1+w1], cur[y1:y1+h1, x1:x1+w1],
                                  cfg.global_min_dist, cfg.global_max_ratio):
            if mine: return "not coherent", info
            continue
        if gate(f1[y1:y1+h1, x1:x1+w1, :]) == 1:
            return ("FOUND" if mine else "other box won first"), info
        if mine: return "LeNet gate rejects", info
    return "?", info

cap = cv2.VideoCapture("data/processed/SOFA-O4/videos/first_catch.avi")
ports = {name: vendor.import_motion_port(gate, cfg, hud_mask=hud) for name, cfg in [("upstream", UPSTREAM), ("clutter", CLUTTER)]}
cfgs = {"upstream": UPSTREAM, "clutter": CLUTTER}
res = collections.defaultdict(list)
prev, n = None, 0
while True:
    ok, fr = cap.read()
    if not ok: break
    n += 1
    if prev is not None and n in boxes:
        for name in ports:
            stage, info = diagnose(ports[name], cfgs[name], prev, fr, boxes[n])
            res[name].append((n, stage, info))
    prev = fr
json.dump(res, open(sys.argv[1], "w"))
for name, rows in res.items():
    print(f"\n{name}")
    for lo, hi in [(708, 800), (801, 900), (901, 953), (954, 964)]:
        c = collections.Counter(s for f, s, _ in rows if lo <= f <= hi)
        print(f"  {lo}-{hi}: {dict(c.most_common())}")
