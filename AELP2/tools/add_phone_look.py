#!/usr/bin/env python3
from __future__ import annotations
"""
Apply a subtle smartphone look: grain, slight vignette, gentle motion jitter.
Usage: python3 AELP2/tools/add_phone_look.py input.mp4 output.mp4
"""
import sys, shutil, subprocess

FFMPEG = shutil.which('ffmpeg') or 'ffmpeg'

def main():
    if len(sys.argv) < 3:
        print('usage: add_phone_look.py in.mp4 out.mp4'); return
    src, dst = sys.argv[1], sys.argv[2]
    # jitters via perspective shift (tiny), noise, vignette
    filt = (
        "noise=alls=8:allf=t+u,"
        "vignette=PI/6,"
        "zoompan=z='min(zoom+0.0005,1.01)':x='iw*0.005*sin(2*PI*n/60)':y='ih*0.005*cos(2*PI*n/75)':d=1:s=1080x1920"
    )
    cmd = f"{FFMPEG} -y -i '{src}' -vf {filt} -c:v libx264 -crf 20 -pix_fmt yuv420p -movflags +faststart '{dst}'"
    subprocess.run(cmd, shell=True, check=True)
    print({'out': dst})

if __name__=='__main__':
    main()

