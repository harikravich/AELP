#!/usr/bin/env python3
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KB_DIR = ROOT / 'AELP2' / 'knowledge' / 'products'
CAND = ROOT / 'AELP2' / 'outputs' / 'creative_candidates'
OUT = ROOT / 'AELP2' / 'reports' / 'candidates_validation.json'

def load_kb():
    m={}
    for f in KB_DIR.glob('*.json'):
        d=json.loads(f.read_text())
        m[d['product_id']]=d
    return m

def main():
    kb=load_kb()
    issues=[]; total=0
    for f in CAND.glob('*.json'):
        d=json.loads(f.read_text())
        dna=d.get('dna') or {}
        pid=dna.get('product_id')
        p=kb.get(pid)
        if not p:
            continue
        total+=1
        text=' '.join(dna.get('copy_lines') or [])
        # prohibited phrases
        for bad in p.get('prohibited_phrases') or []:
            if bad.lower() in text.lower():
                issues.append({'creative_id': dna.get('id'), 'type':'prohibited_phrase', 'phrase': bad})
        # require disclaimer if claim present
        if dna.get('policy',{}).get('mandatory_disclaimer'):
            pass
        else:
            # if any approved claim text is found, require a disclaimer
            claim_texts=[c['text'] for c in p.get('approved_claims') or []]
            if any(ct in text for ct in claim_texts):
                issues.append({'creative_id': dna.get('id'), 'type':'missing_disclaimer'})
    out={'checked': total, 'issues': issues[:200]}
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps({'checked': total, 'issues': len(issues)}, indent=2))

if __name__=='__main__':
    main()

