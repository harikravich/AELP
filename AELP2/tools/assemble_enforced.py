#!/usr/bin/env python3
from __future__ import annotations
"""
Assemble a 9:16 ad with enforced slot grammar using a manifest + recipe.

Inputs:
- Manifest JSON with explicit slot assets, e.g. AELP2/manifests/ugc_sample1.json
  {
    "recipe": "AELP2/creative/recipes/ugc_pattern_v1.json",
    "slots": {
      "hook": "AELP2/outputs/a_roll/hooks/yt_hook_generic_0.mp4",
      "proof": "AELP2/assets/proof_clips/breach_scan_found.mp4",
      "relief": "AELP2/outputs/a_roll/hooks/yt_hook_generic_2.mp4",
      "endcard": "AELP2/outputs/demo_ads/04_cta.png"
    }
  }

Outputs:
- AELP2/outputs/finals/<manifest_stem>_final.mp4
Also writes a small metadata JSON with slot durations and validation results.
"""
import json, subprocess, shlex
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIN = ROOT / 'AELP2' / 'outputs' / 'finals'

def ff(*args):
    subprocess.run(args, check=True)

def probe_duration(path: Path) -> float:
    try:
        out = subprocess.check_output([
            'ffprobe','-v','error','-show_entries','format=duration','-of','default=noprint_wrappers=1:nokey=1',str(path)
        ]).decode().strip()
        return float(out)
    except Exception:
        return 0.0

def ensure_endcard_png() -> Path:
    # Ensure demo endcard exists via storyboard tool
    ec = ROOT/'AELP2'/'outputs'/'demo_ads'/'04_cta.png'
    if not ec.exists():
        subprocess.run(['python3', str(ROOT/'AELP2'/'tools'/'demo_storyboard.py')], check=True)
    return ec

def build_filter_concat(slots: list[tuple[str, Path, float]], end_png: Path|None) -> tuple[list[str], list[str]]:
    inputs = []
    filters = []
    idx = 0
    labels = []
    for sid, path, dur in slots:
        if path.suffix.lower() in {'.png','.jpg','.jpeg'}:
            inputs += ['-loop','1','-t',f'{dur}','-i',str(path)]
            filters.append(f'[{idx}:v]scale=1080:1920,format=yuv420p,setsar=1[v{idx}]')
            labels.append(f'[v{idx}]')
            idx += 1
        else:
            # trim and scale
            inputs += ['-i', str(path)]
            filters.append(f'[{idx}:v]scale=1080:1920:force_original_aspect_ratio=decrease,'
                           f'pad=1080:1920:(ow-iw)/2:(oh-ih)/2,setsar=1,trim=duration={dur},setpts=PTS-STARTPTS[v{idx}]')
            labels.append(f'[v{idx}]')
            idx += 1
    if end_png is not None:
        inputs += ['-loop','1','-t','3','-i',str(end_png)]
        filters.append(f'[{idx}:v]scale=1080:1920,format=yuv420p,setsar=1[v{idx}]')
        labels.append(f'[v{idx}]')
    # concat all
    filters.append(''.join(labels) + f'concat=n={len(labels)}:v=1:a=0[vout]')
    return inputs, filters

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True, help='Path to manifest JSON')
    p.add_argument('--out', help='Override output mp4 path')
    args = p.parse_args()

    FIN.mkdir(parents=True, exist_ok=True)
    man_path = Path(args.manifest)
    data = json.loads(man_path.read_text())
    recipe_path = Path(data.get('recipe', ROOT/'AELP2'/'creative'/'recipes'/'ugc_pattern_v1.json'))
    recipe = json.loads(recipe_path.read_text())

    # Collect slot assets
    slots = []
    meta = { 'recipe': str(recipe_path), 'slots': {}, 'validations': [] }
    for s in recipe['slots']:
        sid = s['id']
        asset = data['slots'].get(sid)
        if not asset:
            if sid == 'endcard':
                asset = str(ensure_endcard_png())
            else:
                raise SystemExit(f"missing slot asset: {sid}")
        pth = Path(asset)
        # infer duration target (clamp between min/max)
        if pth.suffix.lower() in {'.png','.jpg','.jpeg'}:
            # stills → minimal duration within slot bounds
            dur = max(s['min_s'], min(s['max_s'], 3.0))
        else:
            avail = probe_duration(pth)
            # prefer mid of bounds but not exceeding available
            target = (s['min_s'] + s['max_s'])/2
            dur = max(s['min_s'], min(s['max_s'], avail if avail>0 else target))
        slots.append((sid, pth, float(dur)))
        meta['slots'][sid] = { 'path': str(pth), 'duration': float(dur) }

    # Basic validations
    # - aspect will be coerced; ensure all videos exist
    for sid, pth, _ in slots:
        if not pth.exists():
            meta['validations'].append({'slot': sid, 'ok': False, 'reason': 'missing file'})
    if any(v.get('ok') is False for v in meta['validations']):
        print(json.dumps(meta, indent=2))
        raise SystemExit(2)

    # Build filtergraph
    end_png = None  # endcard provided via slots
    inputs, filters = build_filter_concat(slots, end_png)

    final = Path(args.out) if args.out else (FIN / f"{man_path.stem}_final.mp4")
    cmd = ['ffmpeg','-y', *inputs, '-filter_complex', ';'.join(filters), '-map','[vout]',
           '-r','30','-c:v','libx264','-crf','19','-pix_fmt','yuv420p', str(final)]
    ff(*cmd)

    meta['output'] = str(final)
    (final.with_suffix('.json')).write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))

if __name__ == '__main__':
    main()
