#!/usr/bin/env python3
from __future__ import annotations
"""
Compute lightweight aesthetic/quality proxies (no-ref):
 - Laplacian variance (sharpness)
 - Tenengrad (Sobel) focus measure
 - Colorfulness (Hasler-Süsstrunk)
Outputs: AELP2/reports/aesthetic_features.jsonl
"""
import json
from pathlib import Path
import cv2, numpy as np

ROOT=Path(__file__).resolve().parents[2]
FIN=ROOT/'AELP2'/'outputs'/'finals'
OUT=ROOT/'AELP2'/'reports'/'aesthetic_features.jsonl'

def read_mid_frame(path: Path):
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total>0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, total//2)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None

def colorfulness(img):
    (B, G, R) = cv2.split(img.astype("float"))
    rg = np.absolute(R - G)
    yb = np.absolute(0.5*(R + G) - B)
    rbMean, rbStd = (np.mean(rg), np.std(rg))
    ybMean, ybStd = (np.mean(yb), np.std(yb))
    return float(np.sqrt((rbStd**2) + (ybStd**2)) + 0.3*np.sqrt((rbMean**2) + (ybMean**2)))

def main():
    with OUT.open('w') as f:
        for mp4 in sorted(FIN.glob('*.mp4')):
            fr=read_mid_frame(mp4)
            if fr is None:
                f.write(json.dumps({'file': mp4.name, 'error': 'no_frame'})+'\n'); continue
            gray=cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            lap=float(cv2.Laplacian(gray, cv2.CV_64F).var())
            gx=cv2.Sobel(gray, cv2.CV_64F, 1, 0); gy=cv2.Sobel(gray, cv2.CV_64F, 0, 1)
            ten=float(np.sqrt(gx**2 + gy**2).mean())
            cf=float(colorfulness(fr))
            f.write(json.dumps({'file': mp4.name, 'lap_var': lap, 'tenengrad': ten, 'colorfulness': cf})+'\n')
    print(json.dumps({'out': str(OUT)}, indent=2))

if __name__=='__main__':
    main()

