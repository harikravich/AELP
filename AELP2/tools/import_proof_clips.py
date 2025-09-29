#!/usr/bin/env python3
from __future__ import annotations
"""
Import real screen recordings and cut proof clips for slot B.

Inputs:
- AELP2/assets/proof_raw/*.mp4  (full screen recordings)
- AELP2/assets/proof_manifest.json  (maps named clips to in/out times)

Outputs:
- AELP2/assets/proof_clips/<clip_id>.mp4 (2–4s segments, scaled 1080x1920)
- Prints a small JSON index with durations and paths.

Manifest format example:
{
  "clips": [
    {"id":"breach_scan_found", "src":"ios_scan_session.mp4", "start": 12.3, "end": 16.1},
    {"id":"lock_card", "src":"android_lock_card.mp4", "start": 5.0, "end": 8.2}
  ]
}
"""
import json, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / 'AELP2' / 'assets'
RAW = ASSETS / 'proof_raw'
CLIPS = ASSETS / 'proof_clips'
MANIFEST = ASSETS / 'proof_manifest.json'

def ff(*args):
    subprocess.run(args, check=True)

def ensure_dirs():
    RAW.mkdir(parents=True, exist_ok=True)
    CLIPS.mkdir(parents=True, exist_ok=True)

def seconds(x):
    try:
        return float(x)
    except Exception:
        return 0.0

def still_to_video(still: Path, dest: Path, duration: float=3.0):
    # Create a subtle Ken Burns effect on the still, export as 1080x1920.
    ff('ffmpeg','-y','-loop','1','-t',f'{duration}', '-i', str(still),
       '-filter_complex','scale=1080:1920,format=yuv420p,zoompan=z=pzoom+0.0005:d=125',
       '-r','30','-pix_fmt','yuv420p','-c:v','libx264','-crf','19', str(dest))

def import_proof():
    ensure_dirs()
    index = {"clips": []}
    if MANIFEST.exists():
        data = json.loads(MANIFEST.read_text())
        for item in data.get('clips', []):
            src_str = str(item['src'])
            if src_str.startswith('AELP2/'):
                src = ROOT / src_str
            elif src_str.startswith('../outputs/'):
                src = ROOT / 'AELP2' / src_str[3:]
            else:
                src = RAW / src_str
            if not src.exists():
                # allow images in RAW as fallback
                img = src
                if img.suffix.lower() in {'.png','.jpg','.jpeg'}:
                    dest = CLIPS / f"{item['id']}.mp4"
                    still_to_video(img, dest, max(2.0, seconds(item.get('end',3))-seconds(item.get('start',0))))
                    index['clips'].append({"id": item['id'], "path": str(dest), "duration": 3.0})
                    continue
                print(f"warn: missing source {src}")
                continue
            start = seconds(item.get('start',0))
            end = seconds(item.get('end',0))
            if end <= start:
                end = start + 3.0
            duration = max(1.5, min(6.0, end - start))
            out = CLIPS / f"{item['id']}.mp4"
            # Cut and scale to 1080x1920, pad if needed
            ff('ffmpeg','-y','-ss',f'{start}','-i',str(src),'-t',f'{duration}',
               '-vf','scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2',
               '-r','30','-pix_fmt','yuv420p','-c:v','libx264','-crf','19', str(out))
            index['clips'].append({"id": item['id'], "path": str(out), "duration": duration})
    else:
        # If no manifest, try converting any stills into short clips as a stopgap
        for img in sorted(RAW.glob('*.png')) + sorted(RAW.glob('*.jpg')) + sorted(RAW.glob('*.jpeg')):
            out = CLIPS / f"{img.stem}.mp4"
            still_to_video(img, out, 3.0)
            index['clips'].append({"id": img.stem, "path": str(out), "duration": 3.0})
    print(json.dumps(index, indent=2))

if __name__ == '__main__':
    import_proof()
