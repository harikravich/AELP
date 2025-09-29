#!/usr/bin/env python3
"""
Offline creative generation + simulator scoring scaffold.

This tool:
  1) Enumerates product × message × format hypotheses
  2) Generates CreativeDNA JSON specs (copy-first; assets referenced)
  3) Writes candidates to AELP2/outputs/creative_candidates/
  4) (Optional) Calls a simulator scoring hook to produce sim_score and CIs

No live publishing. All outputs are files.
"""
from __future__ import annotations
import os, json, random, hashlib
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'AELP2' / 'outputs' / 'creative_candidates'
OUT.mkdir(parents=True, exist_ok=True)

RNG = random.Random(42)

PRODUCTS = [
    'identity',
    'credit',
    'device_security',
    'family',
    'balance'
]

MESSAGES = [
    'Prevent identity thieves from opening loans',
    'Fast alerts when your info leaks',
    'Protect your family’s online life',
    'Stop scam calls & texts',
    'Secure passwords effortlessly'
]

FORMATS = ['image-1x1', 'image-4x5', 'video-15s', 'video-30s', 'carousel']

INSIGHTS = [
    'Data breach exposure is common — check yours now',
    'Freeze & lock tools reduce fraud risk',
    'Parents want peace of mind — emphasize simplicity',
    'Bundled value beats point tools on cost',
    'Balance parent insights: spot meaningful routine changes (iOS)',
    'Set healthy screen time limits and block harmful content',
]

def mk_id(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:12]

KB_DIR = ROOT / 'AELP2' / 'knowledge' / 'products'

def load_kb(pid: str) -> dict | None:
    p = KB_DIR / f"{pid}.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    return None

def pick_claim(kb: dict) -> tuple[str,str,str]:
    claims = kb.get('approved_claims') or []
    if not claims:
        return (None, None, None)
    c = random.choice(claims)
    return (c['text'], c.get('mandatory_disclaimer') or '', c.get('qualifier') or '')

def pick_cta(kb: dict) -> str:
    ctas = kb.get('approved_ctas') or ['Start Free Trial']
    return random.choice(ctas)

def creative_dna(product_slug: str, message: str, fmt: str, insight: str) -> Dict:
    kb = load_kb(product_slug)
    pname = (kb.get('name') if kb else product_slug)
    claim_text, disclaimer, qualifier = pick_claim(kb) if kb else (None, '', '')
    hook = {
        'image-1x1': 'Your identity was found in a data breach — now what?',
        'image-4x5': 'Stop identity thieves before they start',
        'video-15s': 'In 15 seconds: how to stop identity theft',
        'video-30s': 'What scammers don’t want you to know',
        'carousel': '5 ways Aura protects your family'
    }[fmt]
    cta = pick_cta(kb) if kb else 'Start Free Trial'
    copy = [message, insight]
    if claim_text:
        copy.insert(1, claim_text)
    copy.append(cta)
    # COPPA/FTC-sensitive defaults: parent framing, no targeting kids <13, no unsubstantiated claims
    policy = {
        'coppa': 'No content targeting children under 13; parent/guardian message only',
        'ftc': 'Truthful, substantiated; no unqualified guarantees; see terms for coverage',
        'audio_loudness': '-14 LUFS',
        'caption_legibility': 'contrast >= 4.5:1',
        'mandatory_disclaimer': disclaimer,
        'qualifier': qualifier
    }
    return {
        'id': mk_id('|'.join([product_slug, message, fmt, insight])),
        'product': pname,
        'product_id': product_slug,
        'format': fmt,
        'message': message,
        'insight': insight,
        'hooks': [hook],
        'copy_lines': copy,
        'cta': cta,
        'visuals': {
            'style': 'clean, trustworthy, modern; high contrast captions',
            'logo_safe_zone': True,
            'colors': ['#0A2540', '#00A3FF', '#F5F7FA']
        },
        'policy': policy
    }

def simulate_score(dna: Dict) -> Dict:
    # Placeholder scoring hook (replace with real simulator call).
    base = 0.5
    if 'Family' in dna['product']:
        base += 0.05
    if 'data breach' in ' '.join(dna['copy_lines']).lower():
        base += 0.03
    if dna['format'].startswith('video'):
        base += 0.04
    # Add modest randomness to reflect uncertainty
    score = max(0.0, min(1.0, base + RNG.uniform(-0.05, 0.05)))
    return {
        'sim_score': round(score, 4),
        'ci80_low': round(max(0.0, score - 0.08), 4),
        'ci80_high': round(min(1.0, score + 0.08), 4)
    }

def load_priors():
    p = ROOT / 'AELP2' / 'priors' / 'generator_priors.json'
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}

def main(n:int=200):
    priors = load_priors()
    fmt_weights = None
    if priors.get('formats'):
        # map to our formats roughly
        w = priors['formats']
        # weight video formats higher if 'video' prior is strong
        v = w.get('9x16',0)+w.get('video',0)
        i = w.get('1x1-4x5',0)
        fmt_weights = {
            'video-15s': 0.5 + v,
            'video-30s': 0.5 + v,
            'image-1x1': 0.5 + i,
            'image-4x5': 0.5 + i,
            'carousel': 0.5 + i*0.5
        }
    combos = []
    for _ in range(n):
        p = RNG.choice(PRODUCTS)
        m = RNG.choice(MESSAGES)
        if fmt_weights:
            choices = list(fmt_weights.keys())
            weights = [fmt_weights[k] for k in choices]
            s = sum(weights)
            probs = [w/s for w in weights]
            r = RNG.random(); acc=0.0; f=choices[-1]
            for k,pr in zip(choices, probs):
                acc += pr
                if r <= acc: f=k; break
        else:
            f = RNG.choice(FORMATS)
        i = RNG.choice(INSIGHTS)
        combos.append((p,m,f,i))
    items = []
    for (p,m,f,i) in combos:
        dna = creative_dna(p,m,f,i)
        score = simulate_score(dna)
        items.append({**dna, **score})
        (OUT / f"{dna['id']}.json").write_text(json.dumps({
            'creative_id': dna['id'],
            'dna': dna,
            'score': score
        }, indent=2))
    # Write a bundle summary for ranking
    ranked = sorted(items, key=lambda x: -x['sim_score'])
    (ROOT / 'AELP2' / 'reports' / 'creative_leaderboard.json').write_text(json.dumps({
        'count': len(ranked),
        'items': ranked[:50]
    }, indent=2))
    print(f"Generated {len(items)} candidates → outputs/creative_candidates and reports/creative_leaderboard.json")

if __name__ == '__main__':
    main()
