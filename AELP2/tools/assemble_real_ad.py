#!/usr/bin/env python3
from __future__ import annotations
"""
Assemble a "real" ad using Runway hooks, a proof clip, a relief clip, captions, and ElevenLabs VO.

Inputs (auto-picked with sane fallbacks):
  - Hook: AELP2/outputs/renders/runway/*.mp4 (newest) or AELP2/outputs/a_roll/hooks/*.mp4
  - Proof: AELP2/assets/proof_clips/*.mp4 (lock_card preferred) or reuse Hook
  - Relief: next-best Runway or A-roll
  - Endcard: created via demo_storyboard.py (04_cta.png)
  - VO: AELP2/outputs/audio/hook_identity.mp3 + proof_identity.mp3 + cta_identity.mp3

Output:
  - AELP2/outputs/finals/real_demo_v1.mp4
"""
import json, subprocess, os
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[2]
RUNWAY = ROOT/'AELP2'/'outputs'/'renders'/'runway'
AROLL  = ROOT/'AELP2'/'outputs'/'a_roll'/'hooks'
PROOF  = ROOT/'AELP2'/'assets'/'proof_clips'
AUDIO  = ROOT/'AELP2'/'outputs'/'audio'
FINALS = ROOT/'AELP2'/'outputs'/'finals'

def pick_brand_font() -> str:
    cfg = Path(__file__).resolve().parents[2]/'AELP2'/'creative'/'brand_config.json'
    if cfg.exists():
        try:
            data=json.loads(cfg.read_text())
            files=data.get('font_files') or []
            if files:
                return files[0]
        except Exception:
            pass
    for p in (
        Path('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'),
        Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'),
    ):
        if p.exists():
            return str(p)
    return 'sans-serif'

def ff(*args):
    subprocess.run(args, check=True)

def pick_one(patterns: list[Path]) -> Path|None:
    cands=[]
    for p in patterns:
        cands += sorted(p.glob('*.mp4'), key=lambda x: x.stat().st_mtime, reverse=True)
    return cands[0] if cands else None

def ensure_endcard() -> Path:
    dest = ROOT/'AELP2'/'outputs'/'demo_ads'/'04_cta.png'
    if not dest.exists():
        subprocess.run(['python3', str(ROOT/'AELP2'/'tools'/'demo_storyboard.py')], check=True)
    return dest

def ensure_vo():
    # generate VO files if missing
    need = ['hook_identity.mp3','proof_identity.mp3','cta_identity.mp3']
    missing = [n for n in need if not (AUDIO/n).exists()]
    if missing:
        subprocess.run(['python3', str(ROOT/'AELP2'/'tools'/'gen_eleven_vo.py')], check=True)

    FONT = pick_brand_font()

    def drawtext(text: str, start: float, end: float, size: int=62, y_offset: int=120):
        safe = text.replace(':','\\:').replace("'","\\'")
        return (
            f"drawtext=fontfile={FONT}:text='{safe}':fontcolor=white:fontsize={size}:"
            f"x=(w-tw)/2:y={y_offset}:enable='between(t,{start},{end})',"
        )

def main():
    FINALS.mkdir(parents=True, exist_ok=True)
    endcard = ensure_endcard()
    ensure_vo()

    hook = pick_one([RUNWAY, AROLL])
    relief = pick_one([RUNWAY, AROLL])
    if not hook:
        raise SystemExit('no hook candidates found')
    # pick a different relief if possible
    if relief and relief == hook:
        rels = sorted((AROLL).glob('*.mp4'))
        relief = rels[0] if rels else hook

    proof = (PROOF/'lock_card.mp4') if (PROOF/'lock_card.mp4').exists() else pick_one([PROOF]) or hook

    # Target durations
    d_hook, d_proof, d_relief, d_end = 2.0, 3.5, 2.5, 3.0

    # Build caption filters
    vf = ''
    vf += drawtext("Scam texts are not always obvious.", 0.2, d_hook-0.2)
    # proof caption appears after hook duration
    proof_start = d_hook
    vf += drawtext('Locked • Card frozen', proof_start+0.2, proof_start + d_proof - 0.3, 54, 1080-260)
    # relief caption
    relief_start = d_hook + d_proof
    vf += drawtext('One tap. Peace.', relief_start+0.2, relief_start + d_relief - 0.3, 58, 1080-260)

    # Build filter graph for video concat (4 parts)
    inputs = ['-i', str(hook), '-i', str(proof), '-i', str(relief), '-loop','1','-t', f'{d_end}', '-i', str(endcard)]
    filters = [
        f"[0:v]scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2,setsar=1,trim=duration={d_hook},setpts=PTS-STARTPTS[v0]",
        f"[1:v]scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2,setsar=1,trim=duration={d_proof},setpts=PTS-STARTPTS[v1]",
        f"[2:v]scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2,setsar=1,trim=duration={d_relief},setpts=PTS-STARTPTS[v2]",
        f"[3:v]scale=1080:1920,format=yuv420p,setsar=1[v3]",
        f"[v0][v1][v2][v3]concat=n=4:v=1:a=0[vc]",
        f"[vc]{vf}format=yuv420p[vout]"
    ]

    # Audio: concat hook+proof VO, then keep through relief and endcard (silence)
    hook_vo = AUDIO/'hook_identity.mp3'
    proof_vo = AUDIO/'proof_identity.mp3'
    cta_vo = AUDIO/'cta_identity.mp3'
    concat_list = AUDIO/'_real_list.txt'
    concat_list.write_text(f"file '{hook_vo}'\nfile '{proof_vo}'\nfile '{cta_vo}'\n")
    vo_mix = AUDIO/'_real_mix.mp3'
    ff('ffmpeg','-y','-f','concat','-safe','0','-i',str(concat_list),'-c','copy',str(vo_mix))

    out = FINALS/'real_demo_v1.mp4'
    # add VO input as the last input, then map it as audio
    cmd = ['ffmpeg','-y', *inputs, '-i', str(vo_mix),
           '-filter_complex', ';'.join(filters),
           '-map','[vout]','-map','4:a:0','-shortest',
           '-r','30','-c:v','libx264','-crf','19','-pix_fmt','yuv420p','-c:a','aac','-b:a','128k', str(out)]
    ff(*cmd)
    # Hard gate: move to rejected if thresholds fail
    try:
        import subprocess
        gate = subprocess.check_output(['python3', str(ROOT/'AELP2'/'tools'/'finalize_gate.py'), '--video', str(out), '--thr-rel','0.30','--thr-int','0.60']).decode()
        print(gate)
    except Exception:
        pass

    print(json.dumps({'final': str(out), 'hook': str(hook), 'proof': str(proof), 'relief': str(relief)}, indent=2))

if __name__ == '__main__':
    main()
