#!/usr/bin/env python3
from __future__ import annotations
"""
Ingest Midjourney backplates from a folder, normalize to 9:16, lightly denoise,
write a manifest, and optionally animate via Runway image_to_video.

Usage:
  python3 AELP2/tools/mj_ingest.py --src AELP2/assets/backplates/mj/2025-09-24 --animate

Outputs:
  - Normalized images → AELP2/assets/backplates/normalized/*.jpg
  - If input is an MJ 2x2 grid, auto‑splits into 4 tiles (suffix _u1.._u4) before normalizing.
  - Manifest JSON     → AELP2/assets/backplates/normalized/_manifest.json
  - If --animate      → MP4s under AELP2/outputs/renders/runway/ using each image as promptImage
"""
import json, os, base64, io, subprocess
from pathlib import Path

try:
    import cv2
    OPENCV=True
except Exception:
    OPENCV=False

ROOT = Path(__file__).resolve().parents[2]
NORM = ROOT/'AELP2'/'assets'/'backplates'/'normalized'
RUNWAY_OUT = ROOT/'AELP2'/'outputs'/'renders'/'runway'

def ensure_dirs():
    NORM.mkdir(parents=True, exist_ok=True)
    RUNWAY_OUT.mkdir(parents=True, exist_ok=True)

def center_crop_9x16(img):
    h, w = img.shape[:2]
    target_ratio = 9/16
    cur_ratio = w/h
    if abs(cur_ratio - target_ratio) < 1e-3:
        return img
    # compute crop to match 9:16
    new_w = int(h * target_ratio)
    if new_w <= w:
        x0 = (w - new_w)//2
        return img[:, x0:x0+new_w]
    # else crop height
    new_h = int(w / target_ratio)
    y0 = (h - new_h)//2
    return img[y0:y0+new_h, :]

def is_mj_grid(path: Path, img) -> bool:
    name = path.stem.lower()
    h, w = img.shape[:2]
    if 'grid' in name:
        return True
    # heuristic: square and reasonably large → likely 2x2 grid from MJ
    ar = w/float(h)
    return (0.95 <= ar <= 1.05) and (min(w,h) >= 1024)

def split_grid_2x2(img):
    h, w = img.shape[:2]
    # assume even split; ignore thin gutters
    midx, midy = w//2, h//2
    tiles = [
        img[0:midy, 0:midx],       # u1 (top-left)
        img[0:midy, midx:w],       # u2 (top-right)
        img[midy:h, 0:midx],       # u3 (bottom-left)
        img[midy:h, midx:w],       # u4 (bottom-right)
    ]
    return tiles

def lightly_denoise(img):
    if not OPENCV:
        return img
    try:
        return cv2.fastNlMeansDenoisingColored(img, None, 3, 3, 7, 15)
    except Exception:
        return img

def to_data_uri(img) -> str:
    import PIL.Image as PImage
    im = PImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    im.save(buf, format='JPEG', quality=92)
    b64 = base64.b64encode(buf.getvalue()).decode('ascii')
    return f'data:image/jpeg;base64,{b64}'

def animate_via_runway(data_uri: str, prompt_text: str, out_name: str):
    import requests
    key = os.getenv('RUNWAY_API_KEY')
    assert key, 'RUNWAY_API_KEY missing'
    headers={'Authorization': f'Bearer {key}', 'X-Runway-Version': '2024-11-06'}
    url = 'https://api.runwayml.com/v1/image_to_video'
    payload = {
        'model': 'gen4_turbo',
        'promptImage': data_uri,
        'promptText': f"{prompt_text}. Do not show any on-screen text or UI. Avoid: on-frame text, captions, UI elements, watermark, logo, extra fingers, warped hands",
        'ratio': '720:1280',
        'duration': 4,
        'watermark': False
    }
    s=requests.Session(); s.headers.update(headers)
    r=s.post(url, json=payload, timeout=60)
    r.raise_for_status()
    tid=r.json()['id']
    # poll
    import time
    for _ in range(180):
        jr=s.get(f'https://api.runwayml.com/v1/tasks/{tid}', timeout=30)
        jr.raise_for_status(); js=jr.json()
        st=js.get('status') or js.get('state')
        if st in ('SUCCEEDED','COMPLETED'):
            # find URL
            video=None
            for k in ('output','assets','result','data'):
                v=js.get(k)
                if isinstance(v, dict):
                    video=v.get('video') or v.get('url')
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    video=v[0].get('url') or v[0].get('video')
                if video:
                    break
            if video:
                # download
                with s.get(video, stream=True, timeout=120) as resp:
                    resp.raise_for_status()
                    with open(RUNWAY_OUT/out_name, 'wb') as f:
                        for chunk in resp.iter_content(1<<14):
                            if chunk:
                                f.write(chunk)
            return out_name
        if st in ('FAILED','CANCELED','ERROR'):
            return None
        time.sleep(5)
    return None

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='Folder with MJ exports (jpg/png)')
    ap.add_argument('--animate', action='store_true', help='Animate each normalized image via Runway')
    args = ap.parse_args()

    ensure_dirs()
    src = Path(args.src)
    imgs = sorted([p for p in src.glob('**/*') if p.suffix.lower() in ('.jpg','.jpeg','.png','.webp')])
    if not imgs:
        raise SystemExit('no images found in src')

    manifest={'items': []}
    for p in imgs:
        if not OPENCV:
            raise SystemExit('OpenCV not available for image processing')
        img = cv2.imread(str(p))
        if img is None and p.suffix.lower()=='.webp':
            # convert via ffmpeg in-memory path
            tmp = p.with_suffix('.tmp.png')
            os.system(f"ffmpeg -y -i '{p}' -frames:v 1 '{tmp}' >/dev/null 2>&1")
            if tmp.exists():
                img = cv2.imread(str(tmp))
                try: tmp.unlink()
                except Exception: pass
        if img is None:
            continue
        tiles = [img]
        if is_mj_grid(p, img):
            tiles = split_grid_2x2(img)
        for idx, tile in enumerate(tiles, start=1):
            cropped = center_crop_9x16(tile)
            den = lightly_denoise(cropped)
            # resize to 1080x1920
            norm = cv2.resize(den, (1080,1920), interpolation=cv2.INTER_AREA)
            suffix = f"_u{idx}" if len(tiles)>1 else ""
            out_name = f"mj_{p.stem}{suffix}.jpg"
            out_path = NORM/out_name
            cv2.imwrite(str(out_path), norm, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            tags = []
            name_l = p.stem.lower()
            for k in ('kitchen','couch','desk','park','car','bed','macro','swipe','gesture','lock'):
                if k in name_l:
                    tags.append(k)
            manifest['items'].append({'src': str(p), 'normalized': str(out_path), 'tile': idx if len(tiles)>1 else None, 'tags': tags})

    (NORM/'_manifest.json').write_text(json.dumps(manifest, indent=2))

    if args.animate:
        for it in manifest['items']:
            try:
                img = cv2.imread(it['normalized'])
                uri = to_data_uri(img)
                base = Path(it['normalized']).stem
                out_mp4 = f"{base}_anim.mp4"
                prompt = 'Subtle handheld camera motion, maintain realism, no added text or graphics'
                animate_via_runway(uri, prompt, out_mp4)
            except Exception:
                continue

    print(json.dumps({'normalized_count': len(manifest['items']), 'normalized_dir': str(NORM)}, indent=2))

if __name__=='__main__':
    main()
