#!/usr/bin/env python3
from __future__ import annotations
"""
Build a polished preview ad from your brand + MJ assets without waiting on remote video models.

Steps:
  1) Pick 3 normalized stills under AELP2/assets/backplates/normalized/
  2) Animate each with a subtle Ken Burns pan/zoom locally via ffmpeg (2.0s / 3.5s / 3.0s)
  3) Concatenate + append a brand end-card (demo storyboard)
  4) Overlay a timed quiz band (brand colors/fonts)
  5) Hard-gate (relevance/interestingness) and publish to finals

Output: AELP2/outputs/finals/cool_preview_v1.mp4
"""
import json, subprocess, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NORM = ROOT/'AELP2'/'assets'/'backplates'/'normalized'
REND = ROOT/'AELP2'/'outputs'/'renders'/'local_anim'
FIN  = ROOT/'AELP2'/'outputs'/'finals'

def ff(*args):
    subprocess.run(args, check=True)

def ensure_endcard() -> Path:
    ec = ROOT/'AELP2'/'outputs'/'demo_ads'/'04_cta.png'
    if not ec.exists():
        subprocess.run(['python3', str(ROOT/'AELP2'/'tools'/'demo_storyboard.py')], check=True)
    return ec

def animate_still(img: Path, out: Path, dur: float, direction: str='in', zspeed: float=0.0009):
    out.parent.mkdir(parents=True, exist_ok=True)
    # subtle zoom from 1.00 -> 1.06 over duration, with slight drift
    zstep = (zspeed if direction=='in' else -zspeed)
    zexpr = f"if(eq(on,1),1.00,if(lte(zoom,{1.06 if zstep>0 else 1.00}),zoom+{zstep},zoom))"
    ff('ffmpeg','-y','-loop','1','-i',str(img),
       '-t',f'{dur}','-filter_complex',
       f"scale=1080:1920,zoompan=z='{zexpr}':d={int(30*dur)}:s=1080x1920,format=yuv420p,setsar=1",
       '-r','30','-c:v','libx264','-crf','19','-pix_fmt','yuv420p', str(out))

def build_quiz_pack(start0: float, d0: float, d1: float, d2: float) -> Path:
    pack = {
        'cards': [
            { 'start': round(start0+0.2,2), 'end': round(start0+d0-0.2,2), 'title': 'Which of these is a scam?',
              'options': ['A) Package held — click to reschedule','B) Bank text: “Did you spend $500?”'] },
            { 'start': round(start0+d0+0.1,2), 'end': round(start0+d0+d1-0.2,2), 'title': 'Spot the tell in 2 seconds',
              'options': ['Urgent tone','Login link'] },
            { 'start': round(start0+d0+d1+0.1,2), 'end': round(start0+d0+d1+d2-0.2,2), 'title': 'Protect yourself now',
              'options': ['Freeze cards fast','Check exposures'] }
        ],
        'style': { 'title_size': 66, 'opt_size': 52 }
    }
    p = FIN/'_cool_quiz_pack.json'
    p.write_text(json.dumps(pack, indent=2))
    return p

def pick_normals(n=3) -> list[Path]:
    imgs = sorted([p for p in NORM.glob('*.jpg')])
    if len(imgs) < n:
        # also include png if needed
        imgs = sorted([p for p in NORM.glob('*') if p.suffix.lower() in ('.jpg','.jpeg','.png')])
    if not imgs:
        raise SystemExit('no normalized backplates found; run mj_ingest.py first')
    # biased sampling: prefer names with human content if present
    # Order preference: over-shoulder hand, desk hand, relaxed portrait, coffee shop, macro lock
    prefs = ['over-the-shoulder','desk_scene_hand','relaxed_on_a_couch','relaxed_portrait','coffee_shop','hand','portrait','couch','macro_lock','credit_card','stack_of_mail']
    def score(p: Path):
        s=p.stem.lower();
        sc=0
        for i,kw in enumerate(prefs):
            if kw in s:
                sc += (len(prefs)-i)*10
        return sc
    imgs = sorted(imgs, key=score, reverse=True)
    return imgs[:n]

def main():
    FIN.mkdir(parents=True, exist_ok=True)
    shots = pick_normals(3)
    d0,d1,d2 = 2.0, 3.5, 3.0
    anims=[]
    for i,im in enumerate(shots):
        a = REND/f'shot{i+1}.mp4'
        zspeed = 0.0022 if i==0 else 0.0012
        animate_still(im, a, (d0,d1,d2)[i], direction='in' if i%2==0 else 'out', zspeed=zspeed)
        anims.append(a)
    endcard = ensure_endcard()

    # concat three shots + endcard(3s)
    filters=[
        f"[0:v]setpts=PTS-STARTPTS[v0]",
        f"[1:v]setpts=PTS-STARTPTS[v1]",
        f"[2:v]setpts=PTS-STARTPTS[v2]",
        f"[3:v]scale=1080:1920,format=yuv420p,setsar=1[v3]",
        f"[v0][v1][v2][v3]concat=n=4:v=1:a=0[vout]"
    ]
    out_base = FIN/'cool_preview_v1.mp4'
    ff('ffmpeg','-y','-i',str(anims[0]),'-i',str(anims[1]),'-i',str(anims[2]),'-loop','1','-t','3','-i',str(endcard),
       '-filter_complex',';'.join(filters),'-map','[vout]','-r','30','-c:v','libx264','-crf','19','-pix_fmt','yuv420p', str(out_base))

    # quiz overlay
    pack = build_quiz_pack(0.0,d0,d1,d2)
    out_quiz = FIN/'cool_preview_v1_quiz.mp4'
    subprocess.run(['python3', str(ROOT/'AELP2'/'tools'/'render_quiz_overlay.py'), '--video', str(out_base), '--out', str(out_quiz), '--pack', str(pack)], check=True)

    # gate
    gate = subprocess.check_output(['python3', str(ROOT/'AELP2'/'tools'/'finalize_gate.py'), '--video', str(out_quiz), '--thr-rel','0.30','--thr-int','0.60']).decode()
    print(gate)
    print(json.dumps({'shots': [str(s) for s in shots], 'animated': [str(a) for a in anims], 'final': str(out_quiz)}, indent=2))

if __name__=='__main__':
    main()
