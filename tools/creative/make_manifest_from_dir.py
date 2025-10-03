#!/usr/bin/env python3
from __future__ import annotations
"""
Build a creative manifest CSV from a directory of image files.

Usage:
  python tools/creative/make_manifest_from_dir.py \
    --dir assets/new_creatives \
    --out assets/meta_creatives/ads_manifest.csv

Output columns: ad_id,asset_path,headline,primary_text,destination_url
ad_id is the filename stem by default.
"""
import argparse
from pathlib import Path
import csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    root = Path(args.dir)
    rows = []
    for ext in ('*.jpg','*.jpeg','*.png','*.mp4','*.mov','*.webm','*.mkv','*.avi'):
        for p in root.rglob(ext):
            ad_id = p.stem
            rows.append([ad_id, str(p), '', '', ''])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['ad_id','asset_path','headline','primary_text','destination_url'])
        for r in rows:
            w.writerow(r)
    print(f"Wrote manifest with {len(rows)} assets to {out}")


if __name__ == '__main__':
    main()
