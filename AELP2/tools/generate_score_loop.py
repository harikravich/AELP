#!/usr/bin/env python3
from __future__ import annotations
"""
Orchestrate generate -> feature -> score -> filter loop (stub).
Currently:
  - Scores existing finals via score_new_ads.py
  - Selects Top-K by p_win with LCB>0, writes slate

Outputs: AELP2/reports/topk_slate.json
"""
import json, subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'AELP2'/'reports'/'topk_slate.json'

def main():
    subprocess.run(['python3', str(ROOT/'AELP2'/'tools'/'score_new_ads.py')], check=True)
    scores=json.loads((ROOT/'AELP2'/'reports'/'new_ad_scores.json').read_text())['items']
    passed=[s for s in scores if s['lcb']>0]
    if not passed:
        # fallback: pick top 3 by p_win even if LCB<=0
        passed=sorted(scores, key=lambda r: r['p_win'], reverse=True)[:3]
    slate={'items': passed, 'note': 'LCB>0 required; fallback picks top p_win if none'}
    OUT.write_text(json.dumps(slate, indent=2))
    print(json.dumps({'selected': len(passed), 'out': str(OUT)}, indent=2))

if __name__=='__main__':
    main()

