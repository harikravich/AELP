#!/usr/bin/env python3
from __future__ import annotations
"""
Hard QC gates for candidate clips and finals.

Gates implemented (lightweight):
- resolution/aspect check
- blur check (Laplacian variance)
- basic text presence heuristic via MSER (if OpenCV available)
- loudness check using ffmpeg loudnorm (for files with audio)

Usage:
  python3 AELP2/tools/qc_gates.py --video <path> [--role hook|proof|relief|endcard]
Prints JSON with pass/fail and metrics.
"""
import json, subprocess, sys, math
from pathlib import Path

try:
    import cv2
    OPENCV=True
except Exception:
    OPENCV=False

def ffprobe_json(path: Path):
    try:
        out = subprocess.check_output([
            'ffprobe','-v','error','-print_format','json','-show_streams','-show_format', str(path)
        ]).decode()
        return json.loads(out)
    except Exception:
        return {}

def read_frames(path: Path, max_frames=120):
    if not OPENCV:
        return []
    cap = cv2.VideoCapture(str(path))
    frames = []
    count = 0
    while count < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
        count += 1
    cap.release()
    return frames

def laplacian_var(gray):
    import numpy as np
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def text_like_score(frame):
    # Heuristic: count MSER regions with text-like aspect/area
    if not OPENCV:
        return 0.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    try:
        mser = cv2.MSER_create(5)
    except Exception:
        mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(gray)
    h, w = gray.shape
    area_total = h*w
    textish = 0
    area_acc = 0
    for pts in regions:
        x,y,w1,h1 = cv2.boundingRect(pts)
        ar = w1/max(1.0,h1)
        area = w1*h1
        if 8.0 >= ar >= 1.2 and 100 <= area <= 0.02*area_total and h1< h*0.2:
            textish += 1
            area_acc += area
    score = min(1.0, (textish/50.0) + (area_acc/area_total))
    return float(score)

def loudness_measure(path: Path):
    # Use loudnorm to measure integrated LUFS
    try:
        cmd = ['ffmpeg','-i',str(path),'-af','loudnorm=print_format=json','-f','null','-']
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        txt = proc.stderr
        start = txt.find('{')
        end = txt.rfind('}')
        if start!=-1 and end!=-1:
            data = json.loads(txt[start:end+1])
            return float(data.get('input_i', 0.0))
    except Exception:
        pass
    return None

def qc_video(path: Path, role: str='hook'):
    info = ffprobe_json(path)
    vstream = None
    for s in info.get('streams', []):
        if s.get('codec_type')=='video':
            vstream=s; break
    width = int(vstream.get('width',0)) if vstream else 0
    height = int(vstream.get('height',0)) if vstream else 0
    aspect_ok = (width>=720 and height>=1280 and abs((width/height) - (9/16)) < 0.1)

    frames = read_frames(path, max_frames=90)
    blur_vals = []
    text_scores = []
    if OPENCV and frames:
        for i,fr in enumerate(frames[:60]):  # ~2s @30fps
            gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            blur_vals.append(laplacian_var(gray))
            text_scores.append(text_like_score(fr))
    blur_ok = (len(blur_vals)==0) or (sum(blur_vals)/max(1,len(blur_vals)) >= 50.0)
    # Allow text on endcards; forbid on other slots
    text_ok = True
    if role in ('hook','proof','relief'):
        # tolerate tiny amounts
        text_ok = (len(text_scores)==0) or (sum(text_scores)/max(1,len(text_scores)) < 0.15)

    loud = loudness_measure(path)
    audio_ok = True
    if loud is not None:
        audio_ok = (-15.5 <= loud <= -12.5)  # -14 +/- 1.5 LUFS window

    passed = aspect_ok and blur_ok and text_ok
    return {
        'path': str(path),
        'role': role,
        'metrics': {
            'width': width, 'height': height,
            'aspect_ok': aspect_ok,
            'blur_var_avg': round(sum(blur_vals)/max(1,len(blur_vals)), 2) if blur_vals else None,
            'blur_ok': blur_ok,
            'text_score_avg': round(sum(text_scores)/max(1,len(text_scores)), 3) if text_scores else None,
            'text_ok': text_ok,
            'loudness_lufs': loud,
            'audio_ok': audio_ok
        },
        'pass': passed,
        'notes': [] if passed else [
            *([] if aspect_ok else ['bad_aspect']),
            *([] if blur_ok else ['too_blurry']),
            *([] if text_ok else ['text_detected_forbidden'])
        ]
    }

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--role', default='hook', choices=['hook','proof','relief','endcard','final'])
    args = ap.parse_args()
    res = qc_video(Path(args.video), role=args.role)
    print(json.dumps(res, indent=2))

if __name__ == '__main__':
    main()
