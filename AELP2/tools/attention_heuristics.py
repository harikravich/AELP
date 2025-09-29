#!/usr/bin/env python3
from __future__ import annotations
"""
First-3s attention heuristics (rough):
 - assume captions present
 - estimate scene pace from keyframe interval (ffprobe)
 - estimate on-screen text presence (OCR skipped -> heuristic = captions=True)
Outputs: AELP2/reports/attention_scores.json
"""
import json, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIN = ROOT / 'AELP2' / 'outputs' / 'finals'
OUT = ROOT / 'AELP2' / 'reports' / 'attention_scores.json'

def ffprobe_kf(path: Path) -> float:
    cmd=['ffprobe','-select_streams','v:0','-show_frames','-show_entries','frame=key_frame,pkt_pts_time','-of','json',str(path)]
    p=subprocess.run(cmd, capture_output=True, text=True, check=True)
    js=json.loads(p.stdout)
    times=[float(f['pkt_pts_time']) for f in js.get('frames',[]) if f.get('key_frame')==1 and f.get('pkt_pts_time')]
    if len(times)<2: return 3.0
    gaps=[b-a for a,b in zip(times,times[1:])]
    return sum(gaps[:3])/min(3,len(gaps))

def main():
    rows=[]
    for fp in FIN.glob('*.mp4'):
        try:
            kf_gap=ffprobe_kf(fp)
        except Exception:
            kf_gap=3.0
        score=max(0.0, min(1.0, (1.5/kf_gap)))  # faster scene changes -> higher score
        rows.append({'file': str(fp), 'kf_gap_s': round(kf_gap,2), 'first3s_score': round(score,2)})
    OUT.write_text(json.dumps({'items': rows}, indent=2))
    print(json.dumps({'scored': len(rows), 'out': str(OUT)}, indent=2))

if __name__=='__main__':
    main()

