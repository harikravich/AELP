#!/usr/bin/env python3
from __future__ import annotations
"""
Propose non-claim benefit lines for each product KB from merged copy bank (Meta/Google/Impact/YouTube).
Writes AELP2/reports/kb_non_claim_suggestions.json for HITL review.
"""
import json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MERGED = ROOT / 'AELP2' / 'reports' / 'copy_bank_merged.json'
OUT = ROOT / 'AELP2' / 'reports' / 'kb_non_claim_suggestions.json'

def load():
    return json.loads(MERGED.read_text()) if MERGED.exists() else {'items': []}

def is_claim_like(text: str) -> bool:
    # crude filter: contains hard numbers with % or $ or "up to" patterns → likely claim/offer
    t=text.lower()
    if re.search(r"\b(up to|save|\d+%|\$\d+|\d+/?mo|per month)\b", t):
        return True
    return False

def main():
    merged=load()
    items=merged.get('items') or []
    # Keep short benefit statements; drop price/offer lines and obvious claims
    picks=[it for it in items if 20 <= len(it['text']) <= 140 and not is_claim_like(it['text'])]
    # simple ranking by count
    picks=sorted(picks, key=lambda x: -x.get('count',0))[:200]
    OUT.write_text(json.dumps({'suggestions': picks}, indent=2))
    print(json.dumps({'suggestions': len(picks)}, indent=2))

if __name__=='__main__':
    main()

