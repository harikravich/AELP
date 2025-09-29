#!/usr/bin/env python3
from __future__ import annotations
"""
Compute simple motion features from finals using OpenCV optical flow.
Outputs: AELP2/reports/motion_features.jsonl
"""
import subprocess, json
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
FIN=ROOT/'AELP2'/'outputs'/'finals'
OUT=ROOT/'AELP2'/'reports'/'motion_features.jsonl'

def fps_of(path: Path) -> float:
    cmd=['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=r_frame_rate','-of','default=nokey=1:noprint_wrappers=1',str(path)]
    p=subprocess.run(cmd, capture_output=True, text=True)
    s=p.stdout.strip()
    if '/' in s:
        a,b=s.split('/')
        try:
            return float(a)/float(b)
        except Exception:
            return 30.0
    try:
        return float(s)
    except Exception:
        return 30.0

def motion_stats(path: Path):
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    f0=None; mags=[]
    fps=fps_of(path); max_frames=int(min(8.0, 0.5*cap.get(cv2.CAP_PROP_FRAME_COUNT)/max(1,fps))*fps)
    count=0
    while True:
        ret, frame = cap.read()
        if not ret or count>max_frames:
            break
        gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if f0 is None:
            f0=gray; count+=1; continue
        flow=cv2.calcOpticalFlowFarneback(f0, gray, None, pyr_scale=0.5, levels=2, winsize=15, iterations=2, poly_n=5, poly_sigma=1.1, flags=0)
        mag,_=cv2.cartToPolar(flow[...,0], flow[...,1])
        mags.append(float(np.mean(mag)))
        f0=gray; count+=1
    cap.release()
    if not mags:
        return {'motion_mean': 0.0, 'motion_std': 0.0}
    return {'motion_mean': float(np.mean(mags)), 'motion_std': float(np.std(mags))}

def main():
    n=0
    with OUT.open('w') as f:
        for mp4 in sorted(FIN.glob('*.mp4')):
            m=motion_stats(mp4) or {'motion_mean': None, 'motion_std': None}
            f.write(json.dumps({'file': mp4.name, **m})+'\n')
            n+=1
    print(json.dumps({'files': n, 'out': str(OUT)}, indent=2))

if __name__=='__main__':
    main()

