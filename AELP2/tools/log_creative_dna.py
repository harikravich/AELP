#!/usr/bin/env python3
from __future__ import annotations
"""
Write CreativeDNA + cost placeholders for produced finals; load to BigQuery if configured.
"""
import json, os, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIN = ROOT / 'AELP2' / 'outputs' / 'finals'
OUT = ROOT / 'AELP2' / 'reports' / 'creative_render_log.json'

def main():
    rows=[]
    for fp in FIN.glob('*.mp4'):
        rows.append({
            'path': str(fp),
            'created_at': int(fp.stat().st_mtime),
            'features': {
                'model': 'youtube+aura_site_assembly',
                'aspect': '9:16',
                'duration_s': 15,
                'hook_type': 'theme_based',
                'captions': True,
                'voice_profile': None,
                'proof_device': 'overlay_image',
                'cta': 'theme_default'
            },
            'costs': {
                'gen_video': 0,
                'gen_cost_usd': 0.0,
                'tts_chars': 0,
                'tts_cost_usd': 0.0
            }
        })
    OUT.write_text(json.dumps({'items': rows}, indent=2))
    print(json.dumps({'logged': len(rows), 'out': str(OUT)}, indent=2))

if __name__=='__main__':
    main()
