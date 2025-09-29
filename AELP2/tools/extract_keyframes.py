#!/usr/bin/env python3
from __future__ import annotations
"""
Extract simple keyframes for local MP4s (finals) to AELP2/reports/keyframes/<stem>/tXXXX.jpg.
"""
import subprocess, os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
FIN=ROOT/'AELP2'/'outputs'/'finals'
OUT=ROOT/'AELP2'/'reports'/'keyframes'

def run():
    OUT.mkdir(parents=True, exist_ok=True)
    for mp4 in sorted(FIN.glob('*.mp4')):
        stem=mp4.stem
        d=OUT/stem
        d.mkdir(parents=True, exist_ok=True)
        # Extract 1 fps up to 8 seconds (limit IO)
        cmd=['ffmpeg','-y','-i',str(mp4),'-vf','fps=1,scale=540:-2','-t','8','-q:v','3',str(d/'t%04d.jpg')]
        subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(str(d))

if __name__=='__main__':
    run()

