#!/usr/bin/env python3
from __future__ import annotations
"""
Create a simple Advantage+ Creative manifest from finals.
Writes AELP2/outputs/advantage_manifest.json
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIN = ROOT / 'AELP2' / 'outputs' / 'finals'
OUT = ROOT / 'AELP2' / 'outputs' / 'advantage_manifest.json'

def main():
    items=[]
    for fp in sorted(FIN.glob('*.mp4')):
        items.append({
            'file': str(fp),
            'placement': 'Reels',
            'aspect': '9:16',
            'cta': 'Start Free Trial',
            'disclaimer': 'No one can prevent all identity theft or cybercrime.'
        })
    OUT.write_text(json.dumps({'items': items}, indent=2))
    print(json.dumps({'count': len(items), 'out': str(OUT)}, indent=2))

if __name__=='__main__':
    main()

