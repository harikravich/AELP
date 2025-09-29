#!/usr/bin/env python3
from __future__ import annotations
import json, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AROLL = ROOT / 'AELP2' / 'outputs' / 'a_roll' / 'hooks'
PROOF = ROOT / 'AELP2' / 'outputs' / 'proof_images'
FIN   = ROOT / 'AELP2' / 'outputs' / 'finals'
PATKB = ROOT / 'AELP2' / 'competitive' / 'pattern_to_kb.json'

FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'

def ff(*args):
    subprocess.run(args, check=True)

def drawtext_filter(text: str, start: float, end: float, size: int=68):
    return f"drawtext=fontfile={FONT}:text='{text}':fontcolor=white:fontsize={size}:x=(w-tw)/2:y=160:enable='between(t,{start},{end})',"

def main():
    FIN.mkdir(parents=True, exist_ok=True)
    mapping=json.loads(PATKB.read_text())
    # Select available assets
    hooks=sorted(AROLL.glob('yt_hook_generic_*.mp4'))[:4]
    proofs=sorted(PROOF.glob('proof_*.png')) + sorted(PROOF.glob('proof_*.jpg'))
    if not hooks or not proofs:
        print('{"error":"missing hooks or proofs"}')
        return
    themes=[('did_you_spend','Did you spend $482?','Lock card'),
            ('spot_the_tell','Which text is fake?','Report scam'),
            ('pause_peace','One tap. Peace.','Pause Internet'),
            ('two_screens','Chaos → Control','Lock/Remove')]
    out_files=[]
    for i,(slug,hook_txt,proof_txt) in enumerate(themes):
        base=hooks[i % len(hooks)]
        proof=proofs[i % len(proofs)]
        tmp=FIN/f'_tmp_{slug}.mp4'
        final=FIN/f'{slug}_final.mp4'
        # Overlay captions (hook 0-3s, proof text 5-8s) and proof image at 5-8s
        dt1=drawtext_filter(hook_txt,0,3)
        dt2=drawtext_filter(proof_txt,5,8,56)
        vf=dt1+dt2+f"overlay=(W-w)/2:(H-h)/2:enable='between(t,5,8)',format=yuv420p"
        ff('ffmpeg','-y','-i',str(base),'-loop','1','-i',str(proof),'-filter_complex',vf,'-t','12','-r','30','-c:v','libx264','-crf','19','-pix_fmt','yuv420p',str(tmp))
        # Append end-card from existing storyboard (creates 04_cta.png)
        subprocess.run(['python3', str(ROOT/'AELP2'/'tools'/'demo_storyboard.py')], check=True)
        endcard = ROOT/'AELP2'/'outputs'/'demo_ads'/'04_cta.png'
        ff('ffmpeg','-y','-i',str(tmp),'-loop','1','-t','3','-i',str(endcard),
           '-filter_complex','[1:v]scale=1080:1920,format=yuv420p[v1];[0:v][v1]concat=n=2:v=1:a=0[v]',
           '-map','[v]','-r','30','-c:v','libx264','-crf','19','-pix_fmt','yuv420p',str(final))
        out_files.append(str(final))
    print(json.dumps({'finals': out_files}, indent=2))

if __name__=='__main__':
    main()

