#!/usr/bin/env python3
from __future__ import annotations
"""
Extract LUFS using ffmpeg loudnorm analyze mode for finals.
Outputs: AELP2/reports/audio_lufs.jsonl
"""
import json, subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
FIN=ROOT/'AELP2'/'outputs'/'finals'
OUT=ROOT/'AELP2'/'reports'/'audio_lufs.jsonl'

def loudness_measure(path: Path):
    try:
        cmd=['ffmpeg','-i',str(path),'-af','loudnorm=print_format=json','-f','null','-']
        p=subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        txt=p.stderr; i0=txt.find('{'); i1=txt.rfind('}')
        if i0!=-1 and i1!=-1:
            data=json.loads(txt[i0:i1+1])
            return float(data.get('input_i', 0.0))
    except Exception:
        pass
    return None

def main():
    with OUT.open('w') as f:
        for mp4 in sorted(FIN.glob('*.mp4')):
            lufs=loudness_measure(mp4)
            f.write(json.dumps({'file': mp4.name, 'lufs': lufs})+'\n')
    print(json.dumps({'out': str(OUT)}, indent=2))

if __name__=='__main__':
    main()

