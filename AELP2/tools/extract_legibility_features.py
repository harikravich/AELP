#!/usr/bin/env python3
from __future__ import annotations
"""
Compute legibility proxy per final by reusing self_judge (contrast of top band).
Outputs: AELP2/reports/legibility_features.jsonl
"""
import json, subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
FIN=ROOT/'AELP2'/'outputs'/'finals'
OUT=ROOT/'AELP2'/'reports'/'legibility_features.jsonl'

def main():
    with OUT.open('w') as f:
        for mp4 in sorted(FIN.glob('*.mp4')):
            p=subprocess.run(['python3', str(ROOT/'AELP2'/'tools'/'self_judge.py'), '--video', str(mp4)], capture_output=True, text=True)
            try:
                js=json.loads(p.stdout)
                leg=js['scores'].get('legibility')
                f.write(json.dumps({'file': mp4.name, 'legibility': leg})+'\n')
            except Exception:
                f.write(json.dumps({'file': mp4.name, 'error': 'judge_failed'})+'\n')
    print(json.dumps({'out': str(OUT)}, indent=2))

if __name__=='__main__':
    main()

