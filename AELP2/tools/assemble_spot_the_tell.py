#!/usr/bin/env python3
from __future__ import annotations
"""
Assemble a 9:16 "Spot The Tell" video:
- Hook: 5–7s phone-shot (or placeholder) with three SMS bubbles (overlay)
- Proof: Safe Browsing block overlay
- End-card

Output: AELP2/outputs/finals/spot_the_tell_v1.mp4
"""
import shutil, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RENDERS = ROOT / 'AELP2' / 'outputs' / 'renders'
PROOFS = ROOT / 'AELP2' / 'outputs' / 'proof_images'
ENDS = ROOT / 'AELP2' / 'outputs' / 'endcards'
FINALS = ROOT / 'AELP2' / 'outputs' / 'finals'

FFMPEG = shutil.which('ffmpeg') or 'ffmpeg'
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'

def run(cmd: str):
    subprocess.run(cmd, shell=True, check=True)

def main():
    FINALS.mkdir(parents=True, exist_ok=True)
    hook = None
    # Prefer renders that look like hooks; fall back
    for fp in list((RENDERS / 'runway').glob('*.mp4')) + list(RENDERS.glob('*.mp4')):
        hook = fp; break
    proof = PROOFS / 'proof_safe_browsing_block.png'
    if not proof.exists():
        # fallback to any proof
        for p in PROOFS.glob('*.png'):
            proof = p; break
    endcard = ENDS / 'endcard_default.png'
    out = FINALS / 'spot_the_tell_v1.mp4'

    # Compose SMS bubbles with drawbox/drawtext and a proof overlay
    if hook is None:
        src = f"-f lavfi -t 9 -i color=c=0x151515:s=1080x1920"
    else:
        src = f"-i '{hook}'"

    sms1 = "drawbox=x=80:y=260:w=800:h=120:color=0x222222@0.8:t=fill,drawtext=fontfile=%(f)s:text='Bank: Did you try a $497 charge?':fontsize=38:fontcolor=white:x=100:y=300"
    sms2 = "drawbox=x=120:y=420:w=780:h=110:color=0x2a2a2a@0.8:t=fill,drawtext=fontfile=%(f)s:text='Unknown: Click here to secure your account':fontsize=36:fontcolor=white:x=140:y=460"
    sms3 = "drawbox=x=80:y=580:w=820:h=120:color=0x222222@0.8:t=fill,drawtext=fontfile=%(f)s:text='Carrier: New voicemail: \"urgent\"':fontsize=36:fontcolor=white:x=100:y=620"
    hookcap = "drawtext=fontfile=%(f)s:text='Which text is fake?':fontsize=56:fontcolor=white:x=(w-tw)/2:y=80:shadowcolor=0x000000:shadowx=2:shadowy=2"
    fg = (
        "[0:v]scale=1080:1920,setsar=1,"
        + hookcap % {'f': FONT} + ","
        + sms1 % {'f': FONT} + ","
        + sms2 % {'f': FONT} + ","
        + sms3 % {'f': FONT} + "[hook];"
        "[1:v]scale=1000:-1[pf];"
        "[hook][pf]overlay=x=(W-w)/2:y=1460[main]"
    )
    cmd = f"{FFMPEG} -y {src} -loop 1 -i '{proof}' -filter_complex \"{fg}\" -map '[main]' -t 9 -r 30 -pix_fmt yuv420p -c:v libx264 -crf 20 -movflags +faststart '{out}'"
    run(cmd)
    # endcard
    tail = FINALS / 'spot_the_tell_tail.mp4'
    run(f"{FFMPEG} -y -loop 1 -t 2.5 -i '{endcard}' -r 30 -pix_fmt yuv420p -vf scale=1080:1920,setsar=1 '{tail}'")
    concat_list = FINALS / 'stt_concat.txt'
    concat_list.write_text(f"file '{out}'\nfile '{tail}'\n")
    final = FINALS / 'spot_the_tell_final.mp4'
    run(f"{FFMPEG} -y -f concat -safe 0 -i '{concat_list}' -c copy '{final}'")
    print({'final': str(final)})

if __name__=='__main__':
    main()
