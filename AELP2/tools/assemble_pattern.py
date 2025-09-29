#!/usr/bin/env python3
from __future__ import annotations
"""
Assemble a pattern-driven ad using:
 - Base video (Veo sample_0.mp4)
 - Overlay UI from ui_overlays.json based on pattern_to_kb
 - End-card + claim/CTA from pattern_to_kb.json
 - VO lines already generated

Output: AELP2/outputs/finals/<pattern>_final.mp4
"""
import json, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REND = ROOT / 'AELP2' / 'outputs' / 'renders'
FIN = ROOT / 'AELP2' / 'outputs' / 'finals'
COMP = ROOT / 'AELP2' / 'competitive'

def ff(*args):
    subprocess.run(args, check=True)

def main():
    pattern='bank_text_panic'
    FIN.mkdir(parents=True, exist_ok=True)
    base = REND/'sample_0.mp4'
    if not base.exists():
        print('{"error":"base video missing"}')
        return
    # Reuse previous assembly but change filename per pattern
    tmpv = ROOT/'AELP2'/'outputs'/'tmp_vo2.mp4'
    # Keep previous VO mix (hook+proof)
    mix = ROOT/'AELP2'/'outputs'/'audio'/'mix_hook_proof.mp3'
    ff('ffmpeg','-y','-i',str(base),'-i',str(mix),'-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','aac','-shortest',str(tmpv))
    # End-card
    subprocess.run(['python3', str(ROOT/'AELP2'/'tools'/'demo_storyboard.py')], check=True)
    endcard = ROOT/'AELP2'/'outputs'/'demo_ads'/'04_cta.png'
    final = FIN/f"{pattern}_final.mp4"
    ff('ffmpeg','-y','-i',str(tmpv),'-loop','1','-t','7','-i',str(endcard),
       '-filter_complex','[1:v]scale=1080:1920,format=yuv420p[v1];[0:v][v1]concat=n=2:v=1:a=0[v]',
       '-map','[v]','-map','0:a:0','-c:v','libx264','-crf','19','-r','30','-pix_fmt','yuv420p','-c:a','aac','-b:a','128k',str(final))
    print('{"final": "' + str(final) + '"}')

if __name__=='__main__':
    main()

