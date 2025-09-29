#!/usr/bin/env python3
from __future__ import annotations
"""
Assemble a 9:16 split-screen "Two Screens, Two Outcomes" video.

Inputs (optional):
- Left (chaos) and Right (control) hook mp4s under AELP2/outputs/renders/*
  If none found, uses generated placeholders.
- Proof overlay PNG from AELP2/outputs/proof_images/proof_lock_card.png
- End-card from AELP2/outputs/endcards/endcard_default.png

Output: AELP2/outputs/finals/two_screens_v1.mp4
"""
import os, subprocess, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RENDERS = ROOT / 'AELP2' / 'outputs' / 'renders'
PROOFS = ROOT / 'AELP2' / 'outputs' / 'proof_images'
ENDS = ROOT / 'AELP2' / 'outputs' / 'endcards'
FINALS = ROOT / 'AELP2' / 'outputs' / 'finals'

FFMPEG = shutil.which('ffmpeg') or 'ffmpeg'
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'

def pick_first(patterns):
    for pat in patterns:
        for fp in RENDERS.glob(pat):
            return fp
    return None

def run(cmd):
    subprocess.run(cmd, shell=True, check=True)

def assemble():
    FINALS.mkdir(parents=True, exist_ok=True)
    # try to find inputs
    left = pick_first(['runway/*.mp4', '*.mp4'])
    right = pick_first(['*.mp4', 'runway/*.mp4'])
    proof = PROOFS / 'proof_lock_card.png'
    endcard = ENDS / 'endcard_default.png'
    out = FINALS / 'two_screens_v1.mp4'

    # placeholder sources if none
    left_src = str(left) if left else 'color=c=0x202020:s=540x1920:d=9,drawtext=fontfile={f}:text=Chaos:fontsize=72:fontcolor=white:x=(w-tw)/2:y=h/2-100'.format(f=FONT)
    right_src = str(right) if right else 'color=c=0x101010:s=540x1920:d=9,drawtext=fontfile={f}:text=Control:fontsize=72:fontcolor=white:x=(w-tw)/2:y=h/2-100'.format(f=FONT)

    # Build filtergraph
    # 1) Left: scale to 540x1920
    # 2) Right: scale to 540x1920
    # 3) Stack horizontally to 1080x1920
    # 4) Overlay proof on right side near bottom
    # 5) Add captions
    # 6) Concatenate end-card for ~2.5s

    cmd = [
        FFMPEG,
        '-y',
        # inputs: treat placeholders via -f lavfi
        '-f', 'lavfi', '-t', '9', '-i', left_src if left is None else f"-i '{left_src}'",
    ]
    # The above approach complicates quoting; branch small logic instead
    if left is None and right is None:
        cmd = [FFMPEG, '-y',
               '-f','lavfi','-t','9','-i', left_src,
               '-f','lavfi','-t','9','-i', right_src]
    elif left is None and right is not None:
        cmd = [FFMPEG, '-y',
               '-f','lavfi','-t','9','-i', left_src,
               '-i', str(right)]
    elif left is not None and right is None:
        cmd = [FFMPEG, '-y',
               '-i', str(left),
               '-f','lavfi','-t','9','-i', right_src]
    else:
        cmd = [FFMPEG, '-y', '-i', str(left), '-i', str(right)]

    # proof and endcard as inputs
    cmd += ['-loop','1','-i', str(proof if proof.exists() else PROOFS / 'proof_lock_card.png')]
    cmd += ['-loop','1','-i', str(endcard if endcard.exists() else ENDS / 'endcard_default.png')]

    # Filtergraph labels: [0:v]=left, [1:v]=right, [2:v]=proof, [3:v]=end
    drawtext = f"drawtext=fontfile={FONT}:text='Chaos  →  Control':fontsize=48:fontcolor=white:x=(W-tw)/2:y=60:shadowcolor=0x000000:shadowx=2:shadowy=2"
    filtergraph = (
        "[0:v]scale=540:1920,setsar=1[left];"
        "[1:v]scale=540:1920,setsar=1[right];"
        "[left][right]hstack=inputs=2[stk];"
        "[2:v]scale=400:-1[pf];"
        "[stk][pf]overlay=x=1080-420:y=1480[ov1];"
        f"[ov1]{drawtext}[main]"
    )

    cmd += ['-filter_complex', filtergraph, '-map','[main]','-t','9','-r','30','-pix_fmt','yuv420p','-c:v','libx264','-crf','20','-movflags','+faststart', str(out)]
    run(' '.join(cmd))

    # Append endcard (2.5s)
    tail = FINALS / 'two_screens_tail.mp4'
    run(f"{FFMPEG} -y -loop 1 -t 2.5 -i '{endcard}' -r 30 -pix_fmt yuv420p -vf scale=1080:1920,setsar=1 '{tail}'")
    concat_list = FINALS / 'ts_concat.txt'
    concat_list.write_text(f"file '{out}'\nfile '{tail}'\n")
    final = FINALS / 'two_screens_final.mp4'
    run(f"{FFMPEG} -y -f concat -safe 0 -i '{concat_list}' -c copy '{final}'")
    print({'final': str(final), 'inputs': [str(left) if left else 'lavfi', str(right) if right else 'lavfi']})

if __name__=='__main__':
    assemble()
