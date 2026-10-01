"""Tight crops around given frames' detections, to judge what an object actually is."""
import json, sys
import cv2, numpy as np
pred, video, out, span = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
lo, hi = (int(v) for v in span.split("-"))
CELL, COLS, W = 230, 6, 2.6
per = {}
for line in open(pred, encoding="utf-8"):
    if not line.strip(): continue
    r = json.loads(line); i = int(r["image"].rsplit("_",1)[-1].split(".")[0])
    if r["detections"] and lo <= i <= hi: per[i] = r["detections"][0]["bbox"]
keep = sorted(per)[:: max(1, len(per)//(COLS*4))][:COLS*4]
cap, cells, i = cv2.VideoCapture(video), [], 0
while True:
    ok, f = cap.read()
    if not ok or i > max(keep): break
    i += 1
    if i not in keep: continue
    h, w = f.shape[:2]; x1,y1,x2,y2 = per[i]
    side = int(max(34, W*max(x2-x1, y2-y1)))
    left = int(max(0, min(w-side, (x1+x2)/2 - side/2)))
    top  = int(max(0, min(h-side, (y1+y2)/2 - side/2)))
    c = f[top:top+side, left:left+side].copy()
    s = CELL/c.shape[0]
    c = cv2.resize(c, (CELL, CELL), interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(c, (int((x1-left)*s), int((y1-top)*s)),
                     (int((x2-left)*s), int((y2-top)*s)), (0,255,255), 1)
    cv2.putText(c, f"{i}  {int(x2-x1)}x{int(y2-y1)}px", (4, CELL-6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 1, cv2.LINE_AA)
    cells.append(c)
cap.release()
rows=[]
for k in range(0, len(cells), COLS):
    r = cells[k:k+COLS]
    while len(r) < COLS: r.append(np.zeros((CELL,CELL,3), np.uint8))
    rows.append(np.hstack(r))
cv2.imwrite(out, np.vstack(rows))
print(f"wrote {out}: {len(cells)} crops from {span}")
