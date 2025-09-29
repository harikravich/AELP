#!/usr/bin/env python3
from __future__ import annotations
"""
Build lightweight numeric features for creatives using the cached Meta
`creative_objects` JSON files.

Outputs
  - `AELP2/reports/creative_features/creative_features.jsonl`
  - `AELP2/reports/creative_features/index.json`  (creative_id -> file path)

Feature summary (per creative_id)
  * Basic text stats: title/body character & word counts, sentence counts.
  * Link / video coverage, placement flags, publisher platform coverage.
  * Call-to-action signals (count + SIGN_UP / LEARN_MORE / SHOP_NOW / GET_STARTED).
  * Simple identity proxies: creative name length, family hash, video count.

These features feed the weekly pair builder (`build_labels_weekly.py`) and
ultimately the new-ad ranker training pipeline.
"""
import json, hashlib, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COBJ = ROOT / 'AELP2' / 'reports' / 'creative_objects'
OUTD = ROOT / 'AELP2' / 'reports' / 'creative_features'

def family_of(name: str|None) -> str:
    if not name: return ''
    if '_' in name:
        toks=name.split('_'); return '_'.join(toks[:2])
    return name[:16]


def count_words(text: str) -> int:
    tokens=re.findall(r"[\w']+", text)
    return len(tokens)


def count_sentences(text: str) -> int:
    if not text:
        return 0
    parts=re.split(r"[\.!?]+", text)
    return len([p for p in parts if p.strip()])

def feat_from(d: dict) -> dict:
    ad=d.get('ad') or {}; creative=d.get('creative') or {}
    spec=(creative.get('asset_feed_spec') or {})
    titles=spec.get('titles') or []
    bodies=spec.get('bodies') or []
    title=(titles[0].get('text') if titles else '') or ''
    body=(bodies[0].get('text') if bodies else '') or ''
    urls=spec.get('link_urls') or []
    placements=spec.get('asset_customization_rules') or []
    page=((spec.get('object_story_spec') or {}).get('page_id'))
    call_to_actions=spec.get('call_to_action_types') or []
    videos=spec.get('videos') or []

    # placement flags (best-effort)
    p_fb_story=p_fb_reels=p_ig_story=p_ig_reels=0
    publisher_fb=publisher_ig=publisher_msgr=publisher_aud=0
    unique_publishers=set()
    for r in placements:
        cs=r.get('customization_spec') or {}
        fps=cs.get('facebook_positions') or []
        ips=cs.get('instagram_positions') or []
        pubs=cs.get('publisher_platforms') or []
        unique_publishers.update(pubs)
        if 'facebook' in pubs: publisher_fb=1
        if 'instagram' in pubs: publisher_ig=1
        if 'messenger' in pubs: publisher_msgr=1
        if 'audience_network' in pubs: publisher_aud=1
        if 'story' in fps: p_fb_story=1
        if 'facebook_reels' in fps: p_fb_reels=1
        if 'story' in ips: p_ig_story=1
        if 'reels' in ips: p_ig_reels=1

    name = ad.get('name') or creative.get('name') or ''
    fam = family_of(name)
    hash_denom = float(0xFFFFFFFF)
    fam_hash = int(hashlib.md5(fam.encode('utf-8')).hexdigest()[:8], 16) / hash_denom if fam else 0.0

    cta_set={c.upper() for c in call_to_actions}
    features={
        'title_len': len(title),
        'title_word_count': count_words(title),
        'body_len': len(body),
        'body_word_count': count_words(body),
        'body_sentence_count': count_sentences(body),
        'has_link': 1 if (urls and urls[0].get('website_url')) else 0,
        'num_videos': len(videos),
        'p_fb_story': p_fb_story,
        'p_fb_reels': p_fb_reels,
        'p_ig_story': p_ig_story,
        'p_ig_reels': p_ig_reels,
        'publisher_facebook': publisher_fb,
        'publisher_instagram': publisher_ig,
        'publisher_messenger': publisher_msgr,
        'publisher_audience': publisher_aud,
        'placements_count': len(placements),
        'call_to_action_count': len(call_to_actions),
        'cta_sign_up': 1 if 'SIGN_UP' in cta_set else 0,
        'cta_learn_more': 1 if 'LEARN_MORE' in cta_set else 0,
        'cta_shop_now': 1 if 'SHOP_NOW' in cta_set else 0,
        'cta_get_started': 1 if 'GET_STARTED' in cta_set else 0,
        'name_len': len(name),
        'family_hash': fam_hash,
        'page_id_hash': int(hashlib.md5(str(page).encode('utf-8')).hexdigest()[:8], 16) / hash_denom if page else 0.0,
        'publishers_unique': len(unique_publishers),
    }

    return features

def main():
    OUTD.mkdir(parents=True, exist_ok=True)
    out = OUTD / 'creative_features.jsonl'
    idx = {}
    with out.open('w') as f:
        for fp in sorted(COBJ.glob('*.json')):
            try:
                d=json.loads(fp.read_text())
            except Exception:
                continue
            cid=(d.get('ad') or {}).get('id') or fp.stem
            feats=feat_from(d)
            rec={'creative_id': cid, **feats}
            f.write(json.dumps(rec)+'\n')
            idx[cid]=str(out)
    (OUTD/'index.json').write_text(json.dumps(idx, indent=2))
    print(json.dumps({'count': len(idx), 'out': str(out)}, indent=2))

if __name__=='__main__':
    main()
