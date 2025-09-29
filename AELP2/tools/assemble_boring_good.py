#!/usr/bin/env python3
from __future__ import annotations
"""
Assemble 3 "boringly good" Aura variants:
 - Hook: MJ still (normalized) → subtle motion
 - Proof: proof_clips/lock_card.mp4 (or safe_browsing_block.mp4)
 - Relief: MJ still → subtle motion
 - End card: brand demo storyboard
 - Branded captions + VO optional (kept minimal here to ensure legibility)

Outputs to AELP2/outputs/finals/bg_*.mp4 and runs finalize_gate.
"""
import json, subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
NORM=ROOT/'AELP2'/'assets'/'backplates'/'normalized'
PROOF=ROOT/'AELP2'/'assets'/'proof_clips'
FIN=ROOT/'AELP2'/'outputs'/'finals'
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'

def ff(*args):
    subprocess.run(args, check=True)

def ensure_motion(img: Path, seconds: float, out: Path):
    if out.exists():
        return out
    subprocess.run(['python3', str(ROOT/'AELP2'/'tools'/'still_to_motion.py'),
                    '--image', str(img), '--seconds', str(seconds), '--out', str(out)], check=True)
    return out

def ensure_endcard() -> Path:
    ec = ROOT/'AELP2'/'outputs'/'demo_ads'/'04_cta.png'
    if not ec.exists():
        subprocess.run(['python3', str(ROOT/'AELP2'/'tools'/'demo_storyboard.py')], check=True)
    return ec

def drawtext(text: str, start: float, end: float, size: int=60, y="h*0.08"):
    safe=text.replace(':','\\:').replace("'","\\'")
    return f"drawtext=fontfile={FONT}:text='{safe}':fontcolor=white:fontsize={size}:x=(w-tw)/2:y={y}:enable='between(t,{start},{end})',"

def assemble_one(hook: Path, proof: Path, relief: Path, out: Path):
    FIN.mkdir(parents=True, exist_ok=True)
    endcard=ensure_endcard()
    d0,d1,d2,d3 = 2.0, 4.0, 3.0, 3.0
    inputs=['-i', str(hook), '-i', str(proof), '-i', str(relief), '-loop','1','-t', f'{d3}', '-i', str(endcard)]
    filters=[
        # Per-segment text overlays (avoid enable expressions for ffmpeg 4.4 compatibility)
        f"[0:v]scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2,setsar=1,trim=duration={d0},setpts=PTS-STARTPTS[v0a]",
        f"[v0a]drawtext=fontfile={FONT}:text='Scam texts are not always obvious.':fontcolor=white:fontsize=60:x=(w-tw)/2:y=h*0.08[v0]",
        f"[1:v]scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2,setsar=1,trim=duration={d1},setpts=PTS-STARTPTS[v1a]",
        f"[v1a]drawtext=fontfile={FONT}:text='Locked • Card frozen':fontcolor=white:fontsize=56:x=(w-tw)/2:y=h*0.82[v1]",
        f"[2:v]scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2,setsar=1,trim=duration={d2},setpts=PTS-STARTPTS[v2a]",
        f"[v2a]drawtext=fontfile={FONT}:text='Aura':fontcolor=white:fontsize=64:x=(w-tw)/2:y=h*0.82[v2]",
        f"[3:v]scale=1080:1920,format=yuv420p,setsar=1[v3]",
        f"[v0][v1][v2][v3]concat=n=4:v=1:a=0[vc]",
        f"[vc]unsharp=5:5:1.0:5:5:0.0,format=yuv420p[vout]"
    ]
    ff('ffmpeg','-y', *inputs, '-filter_complex',';'.join(filters), '-map','[vout]','-r','30',
       '-c:v','libx264','-crf','19','-pix_fmt','yuv420p', str(out))
    # Hard gate
    gate = subprocess.check_output(['python3', str(ROOT/'AELP2'/'tools'/'finalize_gate.py'), '--video', str(out)]).decode()
    return out, json.loads(gate)

def pick_norm(name_part: str) -> Path:
    picks=[p for p in NORM.glob('*.jpg') if name_part.lower() in p.name.lower()]
    if not picks:
        # fallback to any normalized still
        picks=sorted(NORM.glob('*.jpg'))
    return sorted(picks)[0]

def main():
    # Choose representative stills
    hook1 = pick_norm('Over-the-shoulder')
    relief1 = pick_norm('Relaxed')
    macro = pick_norm('Macro_lock')
    # Create motion clips
    h1 = ensure_motion(hook1, 2.0, ROOT/'AELP2'/'outputs'/'renders'/'runway'/'_tmp_hook1.mp4')
    r1 = ensure_motion(relief1, 3.0, ROOT/'AELP2'/'outputs'/'renders'/'runway'/'_tmp_relief1.mp4')
    # Proof
    proof = (PROOF/'lock_card.mp4') if (PROOF/'lock_card.mp4').exists() else (PROOF/'safe_browsing_block.mp4')
    if not proof.exists():
        raise SystemExit('No proof clip found; please add a 3–5s screen recording to AELP2/assets/proof_raw and run import_proof_clips.py')
    # Assemble three slight variants (swap hook/relief order or macro detail)
    outs=[]
    o1, g1 = assemble_one(h1, proof, r1, FIN/'bg_lock_card_v1.mp4')
    # variant 2: macro as hook, over-shoulder as relief
    h2 = ensure_motion(macro, 2.0, ROOT/'AELP2'/'outputs'/'renders'/'runway'/'_tmp_hook2.mp4')
    o2, g2 = assemble_one(h2, proof, r1, FIN/'bg_lock_card_v2.mp4')
    # variant 3: use over-shoulder hook and macro relief
    r2 = ensure_motion(macro, 3.0, ROOT/'AELP2'/'outputs'/'renders'/'runway'/'_tmp_relief2.mp4')
    o3, g3 = assemble_one(h1, proof, r2, FIN/'bg_lock_card_v3.mp4')
    outs=[{'file': str(o1), 'gate': g1}, {'file': str(o2), 'gate': g2}, {'file': str(o3), 'gate': g3}]
    print(json.dumps({'variants': outs}, indent=2))

if __name__=='__main__':
    main()
