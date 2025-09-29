#!/usr/bin/env python3
from __future__ import annotations
"""
Sort a big "drop" folder of mixed digital assets into the repo structure.

Default source: ~/AELP/uploads (override with --src).

Rules (by extension and simple heuristics):
- Video (.mp4,.mov,.m4v,.webm)           → AELP2/assets/broll/raw/
- Audio (.mp3,.wav,.aac)                  → AELP2/assets/audio/raw/
- Fonts (.ttf,.otf)                       → AELP2/assets/brand/fonts/
- Docs (.pdf,.docx,.pptx,.key,.zip)       → AELP2/assets/brand/guides/
- Images (.jpg,.jpeg,.png,.webp):
    • If path/name contains 'mj' or 'midjourney' → AELP2/assets/backplates/mj/imported/
    • Else                                      → AELP2/assets/backplates/custom/
- App proof screen recordings (heuristic: filenames with 'screen','ios','android','record','proof')
    • Videos matching the above → AELP2/assets/proof_raw/

Writes a log to AELP2/reports/sort_uploads_log.json with moved files.

Usage:
  python3 AELP2/tools/sort_uploaded_assets.py --src ~/AELP/uploads
  (add --dry-run to preview without moving)
"""
import os, json, shutil, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT/'AELP2'/'reports'/'sort_uploads_log.json'

DEST = {
    'broll': ROOT/'AELP2'/'assets'/'broll'/'raw',
    'audio': ROOT/'AELP2'/'assets'/'audio'/'raw',
    'fonts': ROOT/'AELP2'/'assets'/'brand'/'fonts',
    'guides': ROOT/'AELP2'/'assets'/'brand'/'guides',
    'backplates_mj': ROOT/'AELP2'/'assets'/'backplates'/'mj'/'imported',
    'backplates_custom': ROOT/'AELP2'/'assets'/'backplates'/'custom',
    'icons_svg': ROOT/'AELP2'/'assets'/'brand'/'icons'/'svg',
    'design_psd': ROOT/'AELP2'/'assets'/'brand'/'design_psd',
    'vectors_eps_ai': ROOT/'AELP2'/'assets'/'brand'/'vectors',
    'proof_raw': ROOT/'AELP2'/'assets'/'proof_raw',
}

VIDEO_EXT = {'.mp4','.mov','.m4v','.webm'}
AUDIO_EXT = {'.mp3','.wav','.aac','.m4a'}
FONT_EXT  = {'.ttf','.otf'}
DOC_EXT   = {'.pdf','.docx','.pptx','.key','.zip'}
IMG_EXT   = {'.jpg','.jpeg','.png','.webp'}
SVG_EXT   = {'.svg'}
PS_EXT    = {'.psd','.indd'}
VEC_EXT   = {'.ai','.eps'}
WEB_EXT   = {'.html','.css'}

PROOF_HINTS = re.compile(r"(screen|record|ios|android|proof|ui|screenrec|screen-rec)", re.I)

def ensure_dirs():
    for p in DEST.values():
        p.mkdir(parents=True, exist_ok=True)
    (ROOT/'AELP2'/'reports').mkdir(parents=True, exist_ok=True)

def classify(path: Path) -> Path:
    # ignore macOS resource forks and junk
    if path.name.startswith('._') or path.name.lower()=='.ds_store':
        return None
    name = path.name.lower()
    ext = path.suffix.lower()
    if ext in VIDEO_EXT:
        if PROOF_HINTS.search(name):
            return DEST['proof_raw']
        return DEST['broll']
    if ext in AUDIO_EXT:
        return DEST['audio']
    if ext in FONT_EXT:
        return DEST['fonts']
    if ext in DOC_EXT:
        return DEST['guides']
    if ext in SVG_EXT:
        return DEST['icons_svg']
    if ext in PS_EXT:
        return DEST['design_psd']
    if ext in VEC_EXT:
        return DEST['vectors_eps_ai']
    if ext in IMG_EXT:
        # crude MJ hint by folder or name
        pstr = str(path).lower()
        if ('/mj/' in pstr) or ('midjourney' in pstr) or name.startswith('mj_') or ' mj ' in pstr:
            return DEST['backplates_mj']
        return DEST['backplates_custom']
    return None

def unique_dest(dst_dir: Path, fname: str) -> Path:
    base = Path(fname).stem
    ext = Path(fname).suffix
    cand = dst_dir/fname
    i = 2
    while cand.exists():
        cand = dst_dir/f"{base}_{i}{ext}"
        i += 1
    return cand

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=str(Path.home()/'AELP'/'uploads'))
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    src = Path(os.path.expanduser(args.src))
    if not src.exists():
        raise SystemExit(f"source not found: {src}")

    ensure_dirs()
    moved = []
    skipped = []
    for p in src.rglob('*'):
        if not p.is_file():
            continue
        dst_dir = classify(p)
        if not dst_dir:
            skipped.append(str(p))
            continue
        dst = unique_dest(dst_dir, p.name)
        if args.dry_run:
            moved.append({'src': str(p), 'dst': str(dst), 'dry_run': True})
        else:
            try:
                shutil.move(str(p), str(dst))
                moved.append({'src': str(p), 'dst': str(dst)})
            except Exception as e:
                moved.append({'src': str(p), 'dst': str(dst), 'error': str(e)})

    log = {'src': str(src), 'moved_count': len([m for m in moved if 'error' not in m]), 'moved': moved, 'skipped': skipped}
    REPORT.write_text(json.dumps(log, indent=2))
    print(json.dumps({'moved': log['moved_count'], 'skipped': len(skipped), 'report': str(REPORT)}, indent=2))

if __name__=='__main__':
    main()
