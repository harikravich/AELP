#!/usr/bin/env python3
from __future__ import annotations
"""
Normalize vendor exports (BigSpy/PowerAdSpy/SocialPeta) into GAELP creative_objects.

Input: CSV/JSON files in AELP2/vendor_imports
Output: AELP2/reports/creative_objects/vendor_<source>_<id>.json

We only populate the fields required by build_features_from_creative_objects.py:
  creative.asset_feed_spec.{titles,bodies,link_urls,call_to_action_types,videos,asset_customization_rules}

Usage:
  python3 AELP2/tools/import_vendor_meta_creatives.py --src AELP2/vendor_imports
"""
import argparse, csv, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INDIR = ROOT / 'AELP2' / 'vendor_imports'
OUTDIR = ROOT / 'AELP2' / 'reports' / 'creative_objects'


def sniff_source(fp: Path) -> str:
    name = fp.name.lower()
    if 'bigspy' in name:
        return 'bigspy'
    if 'poweradspy' in name or 'powerad' in name:
        return 'poweradspy'
    if 'socialpeta' in name:
        return 'socialpeta'
    return 'vendor'


def read_rows(fp: Path):
    if fp.suffix.lower() == '.csv':
        with fp.open(newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield row
    elif fp.suffix.lower() == '.json':
        try:
            data = json.loads(fp.read_text())
        except Exception:
            return
        if isinstance(data, dict) and 'data' in data and isinstance(data['data'], list):
            data = data['data']
        if isinstance(data, list):
            for row in data:
                if isinstance(row, dict):
                    yield row
    else:
        return


def pick(row: dict, keys: list[str], default: str|None=None):
    for k in keys:
        if k in row and row[k] not in (None, ''):
            return str(row[k])
    return default


def to_creative_object(row: dict) -> dict|None:
    # Identify basics
    ad_id = pick(row, ['ad_id', 'adID', 'id', 'archive_id', 'ad_archive_id'])
    if not ad_id:
        return None
    headline = pick(row, ['headline', 'title', 'ad_title']) or ''
    primary_text = pick(row, ['text', 'primary_text', 'body', 'ad_text']) or ''
    dest = pick(row, ['destination_url', 'landing_url', 'link', 'url', 'final_url'])
    platform_raw = (pick(row, ['platform', 'platforms', 'network'], '') or '').lower()
    media_type = (pick(row, ['media_type', 'type', 'ad_format'], '') or '').lower()
    page_id = pick(row, ['page_id', 'page', 'pageid'])
    cta = pick(row, ['cta', 'call_to_action', 'call_to_action_type'])

    # Build asset_feed_spec
    titles = [{'text': headline}] if headline else []
    bodies = [{'text': primary_text}] if primary_text else []
    link_urls = [{'website_url': dest}] if dest else []
    call_to_action_types = [cta.upper()] if cta else []
    videos = [{}] if ('video' in media_type) else []

    # Publisher/platform mapping
    pubs = []
    if 'facebook' in platform_raw or 'fb' in platform_raw:
        pubs.append('facebook')
    if 'instagram' in platform_raw or 'ig' in platform_raw:
        pubs.append('instagram')
    if not pubs:
        # default to facebook if unspecified
        pubs.append('facebook')
    asset_customization_rules = [{'customization_spec': {'publisher_platforms': pubs}}]

    d = {
        'ad': {
            'id': ad_id,
            'name': headline or primary_text[:32] or f'vendor_{ad_id}',
        },
        'creative': {
            'name': headline or f'vendor_{ad_id}',
            'asset_feed_spec': {
                'titles': titles,
                'bodies': bodies,
                'link_urls': link_urls,
                'call_to_action_types': call_to_action_types,
                'videos': videos,
                'asset_customization_rules': asset_customization_rules,
                'object_story_spec': {'page_id': page_id} if page_id else {},
            },
        },
        '_source': 'vendor',
    }
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=str(INDIR))
    args = ap.parse_args()

    src = Path(args.src)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    n = 0
    for fp in sorted(src.glob('*.*')):
        source = sniff_source(fp)
        for row in read_rows(fp):
            d = to_creative_object(row)
            if not d:
                continue
            ad_id = d['ad']['id']
            out = OUTDIR / f'vendor_{source}_{ad_id}.json'
            try:
                out.write_text(json.dumps(d, indent=2))
                n += 1
            except Exception as e:
                sys.stderr.write(f'warn: failed {ad_id} from {fp.name}: {e}\n')
    print(json.dumps({'imported': n, 'out_dir': str(OUTDIR)}, indent=2))


if __name__ == '__main__':
    main()

