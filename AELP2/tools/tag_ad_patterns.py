#!/usr/bin/env python3
from __future__ import annotations
"""
Tag ads with hook_type, emotion, proof_device, captions_present (heuristics on text),
format/length (unknown -> defaults), brand/claim/CTA (from text cues).
Inputs: AELP2/competitive/ad_items_scored.json
Outputs: AELP2/competitive/ad_items_tagged.json and pattern freqs
"""
import json, re
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'AELP2' / 'competitive' / 'ad_items_scored.json'
OUT = ROOT / 'AELP2' / 'competitive' / 'ad_items_tagged.json'
FREQ = ROOT / 'AELP2' / 'competitive' / 'pattern_freqs.json'

HOOK_PATTERNS = [
  ('bank_text', r'did you spend|charge|statement|bank|card', 'anxiety'),
  ('breach', r'breach|exposed|leaked|data broker', 'vigilance'),
  ('pause_internet', r'pause the internet|screen time|parent', 'control'),
  ('guess_the_scam', r'scam|phish|spoof', 'vigilance'),
  ('before_after', r'before|after|then|now', 'relief')
]

PROOF_CUES = [
  ('lock_card', r'lock|freeze|card'),
  ('credit_lock', r'credit lock|experian'),
  ('safe_browsing', r'safe browsing|block dangerous sites'),
  ('broker_removal', r'removal|data broker'),
]

CTA_CUES = [
  ('start_free_trial', r'free trial|start free'),
  ('get_protected', r'get protected'),
  ('see_plans', r'see plans|see family plans')
]

def tag(txt: str | None):
    t=(txt or '').lower()
    for name,rx,emo in HOOK_PATTERNS:
        if re.search(rx, t):
            return name, emo
    return 'generic', 'neutral'

def find_cue(txt: str | None, cues):
    t=(txt or '').lower()
    for name,rx in cues:
        if re.search(rx, t): return name
    return None

def main():
    data=json.loads(SRC.read_text())['items']
    out=[]
    freqs=defaultdict(Counter)
    for it in data:
        text=' '.join([it.get('title') or '', it.get('body') or ''])
        ht, emo = tag(text)
        proof = find_cue(text, PROOF_CUES)
        cta = find_cue(text, CTA_CUES)
        rec={**it,
             'hook_type': ht,
             'emotion': emo,
             'proof_device': proof,
             'captions_present': True,
             'format': '9:16',
             'length_s': 15,
             'cta': cta}
        out.append(rec)
        freqs['hook_type'][ht]+=1
        freqs['emotion'][emo]+=1
        if proof: freqs['proof_device'][proof]+=1
        if cta: freqs['cta'][cta]+=1
    OUT.write_text(json.dumps({'items': out}, indent=2))
    FREQ.write_text(json.dumps({k: dict(v) for k,v in freqs.items()}, indent=2))
    print(json.dumps({'tagged': len(out), 'freq_out': str(FREQ)}, indent=2))

if __name__=='__main__':
    main()

