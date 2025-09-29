#!/usr/bin/env python3
from __future__ import annotations
"""
Collect claim overlays (text + mandatory disclaimer) and a default CTA from KB for use in end-cards.
Writes AELP2/branding/overlays.json
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KB = ROOT / 'AELP2' / 'knowledge' / 'products'
OUT = ROOT / 'AELP2' / 'branding' / 'overlays.json'

def main():
    overlays=[]
    default_cta="Start Free Trial"
    for fp in KB.glob('*.json'):
        d=json.loads(fp.read_text())
        ctas=d.get('approved_ctas') or []
        if ctas: default_cta=ctas[0]
        for c in d.get('approved_claims') or []:
            overlays.append({
                'product_id': d.get('product_id'),
                'claim_id': c.get('id'),
                'claim_text': c.get('text'),
                'mandatory_disclaimer': c.get('mandatory_disclaimer') or '',
                'cta_text': default_cta
            })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({'overlays': overlays}, indent=2))
    print(json.dumps({'count': len(overlays), 'out': str(OUT)}, indent=2))

if __name__=='__main__':
    main()

