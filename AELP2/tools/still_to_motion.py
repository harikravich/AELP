#!/usr/bin/env python3
from __future__ import annotations
"""
Turn a still image (1080x1920 recommended) into a subtle-motion MP4.
Effects: gentle pan/zoom, optional vignette.

Usage:
  python3 AELP2/tools/still_to_motion.py --image <path> --seconds 3.0 --out <out.mp4>
"""
import subprocess, argparse
from pathlib import Path

def ff(*args):
    subprocess.run(args, check=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--image', required=True)
    ap.add_argument('--seconds', type=float, default=3.0)
    ap.add_argument('--out', required=True)
    args=ap.parse_args()
    img=Path(args.image); out=Path(args.out)
    d=str(max(0.5, args.seconds))
    # Slight overscale + sinusoidal crop drift for a handheld feel
    vf=(
        "scale=1200:2133,"
        "crop=1080:1920:" \
        "x=(in_w-1080)/2+10*sin(2*PI*t/3):" \
        "y=(in_h-1920)/2+6*sin(2*PI*t/2),"
        "format=yuv420p"
    )
    ff('ffmpeg','-y','-loop','1','-i',str(img),'-t',d,'-r','30','-vf',vf,
       '-c:v','libx264','-crf','19','-pix_fmt','yuv420p',str(out))
    print(out)

if __name__=='__main__':
    main()

