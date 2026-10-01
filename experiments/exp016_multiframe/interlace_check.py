"""EXP-016 pre-flight on analog `catch_2`: is the capture interlaced or torn?

KLT assumes a frame is one instant. A 50i/60i analog feed digitised to 30p is two
instants combined -- a moving edge combs between adjacent rows -- and a torn frame
splices two instants at a horizontal seam. Either breaks a tracker quietly, so it is
checked before any tracking number is believed.

Three measurements, all on the raw decoded frames:
  * comb: mean |row_i - (row_{i-1}+row_{i+1})/2| against the same quantity computed on a
    2x vertically decimated copy. Interlace inflates the first and not the second.
  * field energy: mean |even_rows - odd_rows| resampled, vs mean |col_j - col_{j+1}|.
    A progressive frame has similar vertical and horizontal neighbour differences; an
    interlaced one has far larger vertical.
  * tear: per-row-band frame difference. A torn frame shows a step in the row profile.
Also prints the decode rate, which sizes the rest of the experiment.
"""
import sys
import time

import cv2
import numpy as np

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "data/raw/SOFA-ANALOG/videos/catch_2.mp4"
SAMPLE = set(range(480, 600, 8)) | set(range(100, 200, 16))

cap = cv2.VideoCapture(VIDEO)
prev = None
n = 0
t0 = time.time()
rows = []
band_profiles = []
while True:
    ok, fr = cap.read()
    if not ok:
        break
    n += 1
    if n in SAMPLE:
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        vert = np.abs(g[1:-1] - 0.5 * (g[:-2] + g[2:])).mean()
        half = g[::2]
        vert_half = np.abs(half[1:-1] - 0.5 * (half[:-2] + half[2:])).mean()
        horiz = np.abs(g[:, 1:-1] - 0.5 * (g[:, :-2] + g[:, 2:])).mean()
        field = np.abs(g[0::2][: g.shape[0] // 2] - g[1::2][: g.shape[0] // 2]).mean()
        rows.append((n, vert, vert_half, horiz, field))
    if prev is not None and (n - 1) in SAMPLE:
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        d = np.abs(g - prev)
        band_profiles.append((n, [float(b.mean()) for b in np.array_split(d, 12)]))
    prev = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32)
cap.release()
dt = time.time() - t0
print(f"{VIDEO}: {n} frames decoded in {dt:.1f}s = {n / dt:.1f} fps")
print(f"{'frame':>6} {'vert':>7} {'vert/2':>7} {'horiz':>7} {'field':>7} {'v/h':>6}")
for n_, v, vh, h, f in rows:
    print(f"{n_:6d} {v:7.2f} {vh:7.2f} {h:7.2f} {f:7.2f} {v / h:6.2f}")
a = np.array([[v, vh, h, f] for _, v, vh, h, f in rows])
print(f"median vert/horiz = {np.median(a[:, 0] / a[:, 2]):.3f}   "
      f"(>1.4 suggests interlace; ~1.0 progressive)")
print(f"median vert(full)/vert(half-height) = {np.median(a[:, 0] / a[:, 1]):.3f}")
print("\nrow-band frame difference (tear shows as a step between adjacent bands):")
for n_, prof in band_profiles[:8]:
    print(f"{n_:6d} " + " ".join(f"{p:5.1f}" for p in prof))
