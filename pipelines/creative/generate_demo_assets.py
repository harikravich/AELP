#!/usr/bin/env python3
from __future__ import annotations
"""
Generate a small set of demo ad images keyed by ad_id for feature extraction.

Usage:
  python pipelines/creative/generate_demo_assets.py --out-dir assets/demo_creatives --count 12
"""
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np


def make_banner(text: str, size=(600, 400), seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    # random background gradient
    w, h = size
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    c1 = rng.integers(0, 255, size=3)
    c2 = rng.integers(0, 255, size=3)
    for y in range(h):
        t = y / max(1, h - 1)
        arr[y] = (1 - t) * c1 + t * c2
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    # simple rectangle CTA
    cta_w, cta_h = int(w * 0.5), int(h * 0.18)
    cta_x, cta_y = int(w * 0.25), int(h * 0.7)
    draw.rectangle([cta_x, cta_y, cta_x + cta_w, cta_y + cta_h], fill=(255, 255, 255), outline=(0, 0, 0))
    # text
    txt = text
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    try:
        bbox = draw.textbbox((0, 0), txt, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        # Fallback approximate size
        tw, th = (len(txt) * 6, 10)
    draw.text(((w - tw) / 2, (h - th) / 2), txt, fill=(0, 0, 0), font=font)
    draw.text((cta_x + 10, cta_y + 10), "Shop Now", fill=(0, 0, 0), font=font)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--count', type=int, default=12)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(args.count):
        ad_id = f"demo_ad_{i+1:02d}"
        img = make_banner(f"{ad_id}", seed=100 + i)
        p = out / f"{ad_id}.png"
        img.save(p)
        paths.append((ad_id, str(p)))
    # write manifest
    import csv
    man = out / 'ads_manifest.csv'
    with man.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['ad_id', 'asset_path'])
        for ad_id, path in paths:
            w.writerow([ad_id, path])
    print(f"Wrote {len(paths)} demo creatives and manifest at {out}")


if __name__ == '__main__':
    main()
