#!/usr/bin/env python3
from __future__ import annotations
"""
Extract simple audio features via ffprobe:
 - duration, avg bitrate, channels
 - LUFS placeholder (not computed here)
Writes AELP2/reports/audio_features.json
"""
import json, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIN = ROOT / 'AELP2' / 'outputs' / 'finals'
OUT = ROOT / 'AELP2' / 'reports' / 'audio_features.json'

def ffprobe_audio(path: Path):
    cmd=['ffprobe','-v','error','-select_streams','a:0','-show_entries','stream=channels,bit_rate','-show_entries','format=duration','-of','json',str(path)]
    p=subprocess.run(cmd, capture_output=True, text=True, check=True)
    js=json.loads(p.stdout)
    dur=float(js.get('format',{}).get('duration',0.0))
    st=(js.get('streams') or [{}])[0]
    ch=st.get('channels'); br=st.get('bit_rate')
    return {'duration_s': round(dur,2), 'channels': ch, 'bit_rate': int(br) if br else None, 'lufs': None}

def main():
    rows=[]
    for fp in FIN.glob('*.mp4'):
        try:
            rows.append({'file': str(fp), **ffprobe_audio(fp)})
        except Exception as e:
            rows.append({'file': str(fp), 'error': str(e)})
    OUT.write_text(json.dumps({'items': rows}, indent=2))
    print(json.dumps({'files': len(rows), 'out': str(OUT)}, indent=2))

if __name__=='__main__':
    main()

