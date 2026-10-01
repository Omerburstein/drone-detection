import cv2, numpy as np
from src.data.hud_mask import load_mask
hud = load_mask("data/processed/SOFA-O4/hud_mask.png")
cap = cv2.VideoCapture("data/processed/SOFA-O4/videos/first_catch.avi")
cap.set(cv2.CAP_PROP_POS_FRAMES, 599); ok, fr = cap.read()
print(fr.shape, hud.shape, hud.dtype, hud.mean())
v = fr.copy(); v[hud] = (0, 0, 255)
cv2.imwrite("runs/sofa_o4/exp015_normalised_motion/peek_600.jpg", cv2.resize(v, None, fx=0.6, fy=0.6))
g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
print("row means top/bottom", g[:5].mean(), g[-5:].mean(), "col means l/r", g[:, :5].mean(), g[:, -5:].mean())
print("corner 40x40 means", g[:40,:40].mean(), g[:40,-40:].mean(), g[-40:,:40].mean(), g[-40:,-40:].mean())
