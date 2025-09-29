#!/usr/bin/env python3
from __future__ import annotations
"""
Generate synthetic creative blueprints (visual + copy/offer/CTA/placement) as
Meta-like creative_objects and score them with the trained ranker.

Outputs:
  - AELP2/reports/ad_blueprints_top20.json

Notes:
  - Our feature builder captures text length, CTA types, video flag, publisher platforms,
    and placements flags. We encode those explicitly. Semantic copy themes are carried via
    variants of lengths and CTA selection.
"""
import json, itertools
from pathlib import Path
import subprocess, sys

ROOT = Path(__file__).resolve().parents[2]
COBJ = ROOT / 'AELP2' / 'reports' / 'creative_objects'
OUT = ROOT / 'AELP2' / 'reports' / 'ad_blueprints_top20.json'


def combos():
    motifs = ['threat', 'safety', 'device_ui', 'family', 'robocall']
    subjects = ['no_person', 'adult', 'parent_kid']
    palettes = ['trust', 'alert']
    formats = ['static', 'video6s']
    ctas = ['SIGN_UP', 'LEARN_MORE', 'GET_STARTED']
    placements = [
        {'publisher_platforms': ['facebook','instagram'], 'facebook_positions':['feed'], 'instagram_positions':['stream'], 'name':'feed_only'},
        {'publisher_platforms': ['facebook','instagram'], 'facebook_positions':['facebook_reels','story'], 'instagram_positions':['reels','story'], 'name':'reels_stories'},
        {'publisher_platforms': ['facebook','instagram'], 'facebook_positions':['feed','facebook_reels'], 'instagram_positions':['stream','reels'], 'name':'mixed_feed_reels'},
    ]
    for m, s, p, f, c, plc in itertools.product(motifs, subjects, palettes, formats, ctas, placements):
        yield {
            'motif': m,
            'subject': s,
            'palette': p,
            'format': f,
            'cta': c,
            'placement_name': plc['name'],
            'placement_spec': plc,
        }


def blueprint_to_cobj(bp: dict, idx: int) -> dict:
    # Synthesize text length from motif/subject
    title = {
        'threat': 'Another Data Breach? Stay Safe with Aura',
        'safety': 'Smart, Simple Online Safety',
        'device_ui': 'See Threats Stopped in Real Time',
        'family': 'Protect Your Family Online',
        'robocall': 'Block Scam Calls & Texts',
    }[bp['motif']]
    body_map = {
        'threat': 'Instant alerts, dark web scan, credit lock. Try Aura free.',
        'safety': 'Protect identity, money, and devices. 14‑day free trial.',
        'device_ui': 'AI Safe Browsing + antivirus + VPN in one app.',
        'family': 'Parental controls + screen time + social monitoring.',
        'robocall': 'Spam call/text blocker with fraud alerts. Get started.',
    }
    body = body_map[bp['motif']]
    if bp['subject'] == 'parent_kid':
        body += ' Keep kids safe online.'
    if bp['palette'] == 'alert':
        body += ' Act now.'
    # format → video flag
    videos = [{}] if bp['format'].startswith('video') else []
    pubs = bp['placement_spec']['publisher_platforms']
    rules = [{
        'customization_spec': {
            'publisher_platforms': pubs,
            'facebook_positions': bp['placement_spec'].get('facebook_positions', []),
            'instagram_positions': bp['placement_spec'].get('instagram_positions', []),
        }
    }]
    cid = f"bp_{idx:04d}"
    return {
        'ad': {'id': cid, 'name': f"{bp['motif']}_{bp['subject']}_{bp['palette']}_{bp['format']}_{bp['cta']}_{bp['placement_name']}"},
        'creative': {
            'name': f"{bp['motif']} {bp['format']} {bp['cta']}",
            'asset_feed_spec': {
                'titles': [{'text': title}],
                'bodies': [{'text': body}],
                'link_urls': [{'website_url': 'https://buy.aura.com/'}],
                'call_to_action_types': [bp['cta']],
                'videos': videos,
                'asset_customization_rules': rules,
            }
        },
        '_blueprint': bp,
        '_source': 'blueprint'
    }


def run():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    COBJ.mkdir(parents=True, exist_ok=True)
    bps = list(itertools.islice(combos(), 0, 120))  # cap to 120 combos
    cids = []
    for i, bp in enumerate(bps, 1):
        d = blueprint_to_cobj(bp, i)
        (COBJ / f'vendor_blueprint_{d["ad"]["id"]}.json').write_text(json.dumps(d, indent=2))
        cids.append(d['ad']['id'])

    # Build features and score with existing tools
    subprocess.check_call([sys.executable, str(ROOT / 'AELP2' / 'tools' / 'build_features_from_creative_objects.py')])
    subprocess.check_call([sys.executable, str(ROOT / 'AELP2' / 'tools' / 'score_vendor_creatives.py')])

    # Load scores and pick top 20 blueprints
    scores = json.loads((ROOT / 'AELP2' / 'reports' / 'vendor_scores.json').read_text())['items']
    top = [r for r in scores if str(r['creative_id']).startswith('bp_')][:20]
    # Recover metadata
    rows = []
    for r in top:
        cid = r['creative_id']
        meta = json.loads((COBJ / f'vendor_blueprint_{cid}.json').read_text())
        rows.append({'creative_id': cid, **r, 'blueprint': meta.get('_blueprint')})
    OUT.write_text(json.dumps({'count': len(rows), 'items': rows}, indent=2))
    print(json.dumps({'count': len(rows), 'out': str(OUT)}, indent=2))


if __name__ == '__main__':
    run()

