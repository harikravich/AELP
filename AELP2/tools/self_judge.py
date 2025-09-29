#!/usr/bin/env python3
from __future__ import annotations
"""
Self-judging evaluator for creative quality.

Scores:
- interestingness (0-1): motion energy in first ~2s, cut density proxy
- creativity (0-1): color entropy + hue variance (novelty proxy)
- relevance (0-1): phone/hand proxies via edge density near center + skin-tone ratio (very lightweight)
- legibility (0-1): contrast of caption region + font size proxy (if detect text via QC heuristic)
- audio_craft (0-1): closeness to -14 LUFS
- pCTR (0-1): logistic blend of the above (placeholder; replace with calibrated model)

Usage:
  python3 AELP2/tools/self_judge.py --video <path> [--brief "text"]
"""
import json, math, subprocess
from pathlib import Path

try:
    import cv2, numpy as np
    OPENCV=True
except Exception:
    OPENCV=False

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

def motion_energy(frames):
    if not frames:
        return 0.0
    prev = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    acc = 0.0
    for fr in frames[1:60]:  # ~2 seconds at 30fps
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, prev)
        acc += float(diff.mean())
        prev = gray
    # normalize by a rough factor
    score = min(1.0, acc / (60*12.0))
    return score

def color_entropy(frames):
    if not frames:
        return 0.0
    fr = frames[min(10, len(frames)-1)]
    hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
    h = hsv[:,:,0]
    hist = cv2.calcHist([h],[0],None,[32],[0,180])
    p = hist / (hist.sum() + 1e-6)
    ent = -float((p*(np.log(p+1e-9))).sum())  # nat entropy
    # normalize ~ [0,1] with an empirical max ~3.5
    return max(0.0, min(1.0, ent/3.5))

def center_edge_density(frames):
    if not frames:
        return 0.0
    fr = frames[min(10, len(frames)-1)]
    h, w, _ = fr.shape
    cx0, cy0 = int(w*0.35), int(h*0.35)
    cx1, cy1 = int(w*0.65), int(h*0.65)
    roi = fr[cy0:cy1, cx0:cx1]
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(g, 50, 150)
    score = float(edges.mean())/255.0
    return max(0.0, min(1.0, score*2.0))

def skin_ratio(frames):
    if not frames:
        return 0.0
    fr = frames[min(10, len(frames)-1)]
    hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)
    # very rough skin band in HSV
    mask = ((h>0) & (h<25) & (s>40) & (v>40)) | ((h>160) & (s>40) & (v>40))
    ratio = float(mask.mean())
    return max(0.0, min(1.0, ratio*3.0))

def loudness_measure(path: Path):
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

def score_audio(lufs):
    if lufs is None:
        return 0.5
    # gaussian around -14 LUFS, sigma ~ 1.5
    return float(math.exp(-0.5*((lufs + 14.0)/1.5)**2))

def logistic(x):
    return 1.0/(1.0+math.exp(-x))

def evaluate(video: Path, brief: str|None=None):
    frames = read_frames(video, max_frames=120)
    scores = {}
    # Interestingness
    scores['interestingness'] = float(motion_energy(frames))
    # Back-compat alias for any downstream code that expects 'interesting'
    scores['interesting'] = scores['interestingness']
    # Creativity
    scores['creativity'] = color_entropy(frames)
    # Relevance (very rough: edges in center + skin ratio)
    rel = 0.6*center_edge_density(frames) + 0.4*skin_ratio(frames)
    scores['relevance'] = max(0.0, min(1.0, rel))
    # Legibility (proxy): prefer lower text-like score from QC in hook/proof; for final, assume captions exist — estimate contrast in top band
    leg = 0.5
    if OPENCV and frames:
        fr = frames[min(10,len(frames)-1)]
        h,w,_ = fr.shape
        top = fr[int(h*0.08):int(h*0.20), :]
        gray = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY)
        # estimate contrast (Michelson) on top band
        mn, mx = float(gray.min()), float(gray.max())
        contr = (mx - mn)/(mx + mn + 1e-6)
        leg = max(0.0, min(1.0, contr*1.5))
    scores['legibility'] = leg
    # Audio craft
    lufs = loudness_measure(video)
    scores['audio_craft'] = score_audio(lufs)
    # pCTR (placeholder): blend metrics and squash
    z = (
        1.2*scores['interestingness'] +
        0.9*scores['creativity'] +
        1.3*scores['relevance'] +
        1.1*scores['legibility'] +
        1.0*scores['audio_craft'] - 2.5
    )
    scores['pctr'] = logistic(z)
    return {
        'video': str(video),
        'scores': {k: (round(v,3) if isinstance(v,(int,float)) else v) for k,v in scores.items()},
        'aux': {'lufs': lufs}
    }

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--brief', default=None)
    args = ap.parse_args()
    res = evaluate(Path(args.video), args.brief)
    print(json.dumps(res, indent=2))

if __name__ == '__main__':
    main()
