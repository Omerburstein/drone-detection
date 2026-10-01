"""Does common.homography reproduce GLAD's compensate bit-for-bit? And what does a frame cost?"""
import time, cv2, numpy as np
from common import *
from src.algo.glad import vendor
from src.algo.glad.motion import UPSTREAM
port = vendor.import_motion_port(lambda _: 0, UPSTREAM, hud_mask=HUD)
cap = cv2.VideoCapture(VIDEO); cap.set(cv2.CAP_PROP_POS_FRAMES, 848)
_, a = cap.read(); _, b = cap.read()
p, c = port._prepare(a), port._prepare(b)
comp, border, _ = port.compensate(p, c)
H = homography(p, c)
print("max |diff| vs upstream compensate:", np.abs(warp(p, H).astype(int) - comp.astype(int)).max())
dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
t = time.time(); H = homography(p, c); print("H", time.time() - t)
t = time.time(); ms = maps(a, b, H); print("maps no-flow", time.time() - t, len(ms))
t = time.time(); ms = maps(a, b, H, dis); print("maps with DIS", time.time() - t)
v = valid_mask(H, p.shape)
t = time.time(); [peaks(m, v) for m in ms.values()]; print("peaks", time.time() - t)
t = time.time(); cv2.calcOpticalFlowFarneback(warp(prep(a,5),H), prep(b,5), None, 0.5, 4, 21, 5, 7, 1.5, 0); print("farneback", time.time() - t)
