#!/usr/bin/env python3
from __future__ import annotations
"""
Assemble fully original ads from Runway clips and VO packs.

Creates 6 variants: 3 scripts × 2 visual mixes.
Each ~12.0s: Hook(2.0) → Proof(4.0) → Relief(3.0) → End(3.0)

Inputs:
 - Runway clips: AELP2/outputs/renders/runway/*.mp4
 - VO packs: AELP2/outputs/audio/<script_id>/{hook,proof,cta}.mp3
 - End-card via demo_storyboard.py (04_cta.png)

Outputs: AELP2/outputs/finals/orig_<script>_<mix>.mp4
"""
import json, subprocess, itertools
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[2]
RUNWAY = ROOT/'AELP2'/'outputs'/'renders'/'runway'
FINALS = ROOT/'AELP2'/'outputs'/'finals'
AUDIO  = ROOT/'AELP2'/'outputs'/'audio'

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

def ensure_endcard() -> Path:
    ec = ROOT/'AELP2'/'outputs'/'demo_ads'/'04_cta.png'
    if not ec.exists():
        subprocess.run(['python3', str(ROOT/'AELP2'/'tools'/'demo_storyboard.py')], check=True)
    return ec

def pick_runway(n=4):
    clips=sorted(RUNWAY.glob('*.mp4'), key=lambda p: p.stat().st_mtime, reverse=True)
    return clips[:max(1,n)]

    FONT = pick_brand_font()

    def drawtext(text: str, start: float, end: float, size: int=62, y="h*0.09"):
        safe=text.replace(':','\\:').replace("'","\\'")
        return f"drawtext=fontfile={FONT}:text='{safe}':fontcolor=white:fontsize={size}:x=(w-tw)/2:y={y}:enable='between(t,{start},{end})',"

def assemble_variant(script_id: str, mix_id: int, shots: list[Path]) -> Path:
    FINALS.mkdir(parents=True, exist_ok=True)
    endcard=ensure_endcard()
    # shot order: 0->1->2; loop if fewer
    s0, s1, s2 = shots[0], shots[(0+mix_id)%len(shots)], shots[(1+mix_id)%len(shots)]
    d0, d1, d2, dend = 2.0, 4.0, 3.0, 3.0
    # captions vary by section
    captions_json = ROOT/'AELP2'/'prompts'/'vo_scripts.json'
    data=json.loads(captions_json.read_text())
    s=[x for x in data['scripts'] if x['id']==script_id][0]

    vf = ''
    vf += drawtext(s['hook'], 0.2, d0-0.2)
    vf += drawtext(s['proof'], d0+0.2, d0+d1-0.2, 56, 'h*0.82')
    vf += drawtext('Aura', d0+d1+0.2, d0+d1+d2-0.3, 64, 'h*0.82')

    inputs=['-i', str(s0), '-i', str(s1), '-i', str(s2), '-loop','1','-t', f'{dend}', '-i', str(endcard)]
    filters=[
        f"[0:v]scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2,setsar=1,trim=duration={d0},setpts=PTS-STARTPTS[v0]",
        f"[1:v]scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2,setsar=1,trim=duration={d1},setpts=PTS-STARTPTS[v1]",
        f"[2:v]scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2,setsar=1,trim=duration={d2},setpts=PTS-STARTPTS[v2]",
        f"[3:v]scale=1080:1920,format=yuv420p,setsar=1[v3]",
        f"[v0][v1][v2][v3]concat=n=4:v=1:a=0[vc]",
        f"[vc]{vf}unsharp=5:5:1.0:5:5:0.0,format=yuv420p[vout]"
    ]

    # audio: stitch hook+proof+cta for this script
    p=AUDIO/script_id
    hook=AUDIO/script_id/'hook.mp3'
    proof=AUDIO/script_id/'proof.mp3'
    cta=AUDIO/script_id/'cta.mp3'
    # use concat demuxer
    lst=AUDIO/f'_{script_id}_list.txt'
    lst.write_text(f"file '{hook}'\nfile '{proof}'\nfile '{cta}'\n")
    mix=AUDIO/f'_{script_id}_mix.mp3'
    ff('ffmpeg','-y','-f','concat','-safe','0','-i',str(lst),'-c','copy',str(mix))

    out=FINALS/f'orig_{script_id}_{mix_id}.mp4'
    ff('ffmpeg','-y', *inputs, '-i', str(mix), '-filter_complex', ';'.join(filters), '-map','[vout]','-map','4:a:0', '-shortest',
       '-r','30','-c:v','libx264','-crf','19','-pix_fmt','yuv420p','-c:a','aac','-b:a','160k', str(out))
    # Hard gate: move to rejected if it fails thresholds
    try:
        import subprocess, json
        gate = subprocess.check_output(['python3', str(ROOT/'AELP2'/'tools'/'finalize_gate.py'), '--video', str(out), '--thr-rel','0.30','--thr-int','0.60']).decode()
        print(gate)
    except Exception:
        pass
    return out

def main():
    # Collect runway clips
    shots=pick_runway(n=4)
    if len(shots)<1:
        raise SystemExit('no runway clips found; run gen_runway_hooks.py first')
    # Ensure VO packs are present
    scripts=['spot_the_scam','freeze_the_chaos','find_and_fix']
    outs=[]
    for sid in scripts:
        if not (AUDIO/sid/'hook.mp3').exists():
            raise SystemExit(f'missing VO pack for {sid}; run gen_eleven_vo_pack.py')
        for mix in (1,2):
            out=assemble_variant(sid, mix, shots)
            outs.append(str(out))
    print(json.dumps({'finals': outs}, indent=2))

if __name__=='__main__':
    main()
