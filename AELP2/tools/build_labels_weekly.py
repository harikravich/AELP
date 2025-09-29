#!/usr/bin/env python3
from __future__ import annotations
"""
Construct pairwise training data (diff features, label win vs baseline) from
weekly_creatives + creative_features.

Outputs
  AELP2/reports/new_ranker/pairs.jsonl  (fields: cid, week, x[], y, placement)
  AELP2/reports/new_ranker/feature_map.json (name -> index)
"""
import json, math, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WEEKLY = ROOT/ 'AELP2' / 'reports' / 'weekly_creatives'
FEATJL = ROOT/ 'AELP2' / 'reports' / 'creative_features' / 'creative_features.jsonl'
OUTD = ROOT/ 'AELP2' / 'reports' / 'new_ranker'

KEYS = [
    'title_len',
    'title_word_count',
    'body_len',
    'body_word_count',
    'body_sentence_count',
    'has_link',
    'num_videos',
    'p_fb_story',
    'p_fb_reels',
    'p_ig_story',
    'p_ig_reels',
    'publisher_facebook',
    'publisher_instagram',
    'publisher_messenger',
    'publisher_audience',
    'publishers_unique',
    'placements_count',
    'call_to_action_count',
    'cta_sign_up',
    'cta_learn_more',
    'cta_shop_now',
    'cta_get_started',
    'name_len',
    'family_hash',
    'page_id_hash',
]

def load_features():
    idx={}
    if not FEATJL.exists(): return idx
    with FEATJL.open() as f:
        for line in f:
            d=json.loads(line)
            idx[d['creative_id']]=d
    return idx

def actual_cac(it):
    p=float(it.get('actual_score') or 0.0); s=float(it.get('test_spend') or 0.0)
    return (s/p) if p>0 else float('inf')

def utility(it, target):
    p=float(it.get('actual_score') or 0.0); s=float(it.get('test_spend') or 0.0)
    return p - (s/target if target>0 else 0.0)

def feat_vec(feats: dict) -> list[float]:
    return [float(feats.get(k) or 0.0) for k in KEYS]

def main():
    feats=load_features()
    OUTD.mkdir(parents=True, exist_ok=True)
    out = (OUTD / 'pairs.jsonl').open('w')
    fmap = {k:i for i,k in enumerate(KEYS)}
    # group by campaign id
    files=sorted(WEEKLY.glob('*.json'))
    by_c={}
    for f in files:
        m=re.match(r'^(\d+)_([0-9]{4}W[0-9]{2})\.json$', f.name)
        if not m: continue
        cid=m.group(1)
        by_c.setdefault(cid, []).append(f)

    n=0
    for cid, flist in by_c.items():
        weeks=[json.loads(Path(ff).read_text()) for ff in sorted(flist)]
        for i in range(1, len(weeks)):
            cur=weeks[i]
            items=cur.get('items') or []
            cacs=[actual_cac(it) for it in items if math.isfinite(actual_cac(it))]
            target=sorted(cacs)[len(cacs)//2] if cacs else 200.0
            j0=max(0,i-4); j1=i-1
            util_by_cre={}; last={}
            for j in range(j0, j1+1):
                wk=weeks[j]
                for it in (wk.get('items') or []):
                    cr=it.get('creative_id'); u=utility(it, target)
                    util_by_cre[cr]=util_by_cre.get(cr,0.0)+u; last[cr]=it
            if not util_by_cre:
                continue
            base_id=max(util_by_cre.items(), key=lambda kv: kv[1])[0]
            base=last[base_id]; b_p=float(base.get('actual_score') or 0.0); b_cac=actual_cac(base)
            base_feat=feat_vec(feats.get(base.get('creative_id'), {}))
            for v in items:
                cidv=v.get('creative_id'); vf=feat_vec(feats.get(cidv, {}))
                x=[a-b for a,b in zip(vf, base_feat)]
                y=1 if ((float(v.get('actual_score') or 0.0) >= b_p) and (actual_cac(v) <= b_cac)) else 0
                out.write(json.dumps({'campaign': cid, 'week': cur.get('iso_week'), 'x': x, 'y': y})+'\n')
                n+=1
    out.close()
    (OUTD/'feature_map.json').write_text(json.dumps(fmap, indent=2))
    print(json.dumps({'pairs': n, 'features': KEYS, 'out': str(OUTD/'pairs.jsonl')}, indent=2))

if __name__=='__main__':
    main()
