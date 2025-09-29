#!/usr/bin/env python3
"""
Join heuristic CreativeDNA-style features to historical ad items using ad_name and placement mix.

Input: AELP2/reports/creative/*.json
Output: AELP2/reports/creative_with_dna/*.json
"""
from __future__ import annotations
import json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INP = ROOT / 'AELP2' / 'reports' / 'creative'
OUT = ROOT / 'AELP2' / 'reports' / 'creative_with_dna'
OUT.mkdir(parents=True, exist_ok=True)

BALANCE_WORDS = re.compile(r"\b(balance|parent|family|screen time|controls)\b", re.I)
BREACH_WORDS = re.compile(r"\b(breach|leak|identity|fraud)\b", re.I)
SPEED_WORDS = re.compile(r"\b(fast|instant|quick)\b", re.I)

def infer_format(placement_mix: dict) -> str:
    keys = ' '.join(placement_mix.keys()).lower() if placement_mix else ''
    if 'story' in keys or 'stories' in keys or 'reels' in keys:
        return 'vertical-video'
    if 'instream_video' in keys or 'video_feeds' in keys:
        return 'video'
    return 'image-or-feed'

def dna_from(ad_name: str | None, placement_mix: dict) -> dict:
    tags = []
    if ad_name:
        if BALANCE_WORDS.search(ad_name):
            tags.append('balance_parental')
        if BREACH_WORDS.search(ad_name):
            tags.append('breach_alerts')
        if SPEED_WORDS.search(ad_name):
            tags.append('speed_benefit')
    fmt = infer_format(placement_mix)
    return {
        'format': fmt,
        'tags': tags
    }

def main():
    for jf in sorted(INP.glob('*.json')):
        data = json.loads(jf.read_text())
        items = data.get('items') or []
        out_items = []
        for it in items:
            dna = dna_from(it.get('ad_name'), it.get('placement_mix') or {})
            out_items.append({**it, 'dna': dna})
        out = {'campaign_id': data.get('campaign_id'), 'items': out_items}
        (OUT / jf.name).write_text(json.dumps(out, indent=2))
    print(f"Wrote {len(list(OUT.glob('*.json')))} files to {OUT}")

if __name__ == '__main__':
    main()

