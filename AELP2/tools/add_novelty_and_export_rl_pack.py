#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MOD = ROOT / 'AELP2' / 'models' / 'new_ad_ranker'
FEATJL = ROOT / 'AELP2' / 'reports' / 'creative_features' / 'creative_features.jsonl'
BLUE = ROOT / 'AELP2' / 'reports' / 'ad_blueprints_top20.json'
FORE = ROOT / 'AELP2' / 'reports' / 'us_cac_volume_forecasts.json'
RL = ROOT / 'AELP2' / 'reports' / 'rl_test_pack.json'
BRIEF = ROOT / 'AELP2' / 'reports' / 'asset_briefs.json'


def load_features():
    feats = {}
    with FEATJL.open() as f:
        for line in f:
            d = json.loads(line)
            feats[str(d['creative_id'])] = d
    return feats


def main():
    fmap = json.loads((MOD / 'feature_map.json').read_text())
    keys = list(fmap.keys())
    ref = np.asarray(json.loads((MOD / 'feature_ref.json').read_text())['ref'], float)
    feats = load_features()
    blue = json.loads(BLUE.read_text())['items']
    fore = {r['creative_id']: r for r in json.loads(FORE.read_text())['items']}

    items = []
    briefs = []
    for r in blue:
        cid = r['creative_id']
        fv = np.asarray([float(feats[cid].get(k, 0.0)) for k in keys], float)
        dist = float(np.linalg.norm(fv - ref))
        novelty = float(min(1.0, dist / (np.linalg.norm(ref) + 1e-6)))
        bp = r['blueprint']
        # RL config per blueprint
        cfg = {
            'creative_id': cid,
            'motif': bp['motif'],
            'subject': bp['subject'],
            'palette': bp['palette'],
            'format': bp['format'],
            'cta': bp['cta'],
            'placements': bp['placement_spec'],
            'priors': {'p_win': r['p_win'], 'novelty': novelty},
            'forecasts': fore.get(cid, {}).get('budgets', {}),
        }
        items.append(cfg)
        # Asset brief
        briefs.append({
            'creative_id': cid,
            'headline': {
                'threat': 'Another Data Breach? Stay Safe with Aura',
                'safety': 'Smart, Simple Online Safety',
                'device_ui': 'See Threats Stopped in Real Time',
                'family': 'Protect Your Family Online',
                'robocall': 'Block Scam Calls & Texts',
            }[bp['motif']],
            'overlay': bp['palette'] == 'alert',
            'motion': bp['format'].startswith('video'),
            'cta': bp['cta'],
            'placements': bp['placement_spec'],
            'notes': [
                'Use brand palette; ensure logo legible.',
                'Show app UI if device_ui motif; otherwise 1–2 faces max.',
                'Keep text overlays < 20% area; add captions for video.',
            ],
        })

    RL.write_text(json.dumps({'items': items}, indent=2))
    BRIEF.write_text(json.dumps({'items': briefs}, indent=2))
    print(json.dumps({'rl_out': str(RL), 'briefs_out': str(BRIEF), 'count': len(items)}, indent=2))


if __name__ == '__main__':
    main()

