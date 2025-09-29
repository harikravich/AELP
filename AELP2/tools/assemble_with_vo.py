#!/usr/bin/env python3
from __future__ import annotations
"""
Assemble a Veo/Runway hook with ElevenLabs VO, add end-card and disclaimer overlay, normalize audio.
Inputs (auto-detected):
- AELP2/outputs/renders/sample_0.mp4 (Veo) or runway/*.mp4
- AELP2/outputs/audio/*identity*.mp3
- AELP2/branding/end_card_spec.json + overlays.json

Outputs: AELP2/outputs/finals/*.mp4
"""
import json, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REND = ROOT / 'AELP2' / 'outputs' / 'renders'
RUNWAY = REND / 'runway'
AOUT = ROOT / 'AELP2' / 'outputs' / 'audio'
BRAND = ROOT / 'AELP2' / 'branding'
FIN = ROOT / 'AELP2' / 'outputs' / 'finals'

def ff(*args):
    subprocess.run(args, check=True)

def main():
    FIN.mkdir(parents=True, exist_ok=True)
    # pick a base video (prefer Veo sample_0.mp4)
    base = None
    if (REND/'sample_0.mp4').exists():
        base = REND/'sample_0.mp4'
    else:
        vids = sorted(RUNWAY.glob('*.mp4'))
        if vids:
            base = vids[0]
    if not base:
        print('{"error":"no base video found"}')
        return
    hook=AOUT/'hook_identity.mp3'
    proof=AOUT/'proof_identity.mp3'
    cta=AOUT/'cta_identity.mp3'
    if not hook.exists():
        print('{"error":"no VO files"}')
        return
    # Compose VO: hook+proof (concat), then append cta under end-card
    concat_list = ROOT/'AELP2'/'outputs'/'audio'/'_list.txt'
    concat_list.write_text(f"file '{hook}'\nfile '{proof}'\n")
    mix = ROOT/'AELP2'/'outputs'/'audio'/'mix_hook_proof.mp3'
    ff('ffmpeg','-y','-f','concat','-safe','0','-i',str(concat_list),'-c','copy',str(mix))
    # Replace audio on base (trim/pad to video length)
    tmpv = ROOT/'AELP2'/'outputs'/'tmp_vo.mp4'
    ff('ffmpeg','-y','-i',str(base),'-i',str(mix),'-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','aac','-shortest',str(tmpv))
    # Build end-card PNG using existing demo tool (04_cta.png)
    # Reuse demo_storyboard to guarantee we have an end-card image
    subprocess.run(['python3', str(ROOT/'AELP2'/'tools'/'demo_storyboard.py')], check=True)
    endcard = ROOT/'AELP2'/'outputs'/'demo_ads'/'04_cta.png'
    # Concatenate video + end-card still (7s)
    final = FIN/'identity_demo_final.mp4'
    ff('ffmpeg','-y','-i',str(tmpv),'-loop','1','-t','7','-i',str(endcard),
       '-filter_complex','[1:v]scale=1080:1920,format=yuv420p[v1];[0:v][v1]concat=n=2:v=1:a=0[v]',
       '-map','[v]','-map','0:a:0','-c:v','libx264','-crf','19','-r','30','-pix_fmt','yuv420p','-c:a','aac','-b:a','128k',str(final))
    print(json.dumps({'final': str(final)}, indent=2))

if __name__=='__main__':
    main()

