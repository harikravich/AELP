#!/usr/bin/env python3
from __future__ import annotations
import json, sys
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / 'AELP2' / 'reports' / 'runway_tasks.json'
OUTDIR = ROOT / 'AELP2' / 'outputs' / 'renders' / 'runway'

def download(url: str, dest: Path):
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, 'wb') as f:
            for ch in r.iter_content(1<<14):
                if ch: f.write(ch)

def main():
    if not LOG.exists():
        print('no runway_tasks.json'); return
    OUTDIR.mkdir(parents=True, exist_ok=True)
    js=json.loads(LOG.read_text())
    results=js.get('results') or js.get('submitted') or []
    n=0
    for r in results:
        rid=r.get('id') or r.get('taskId') or 'task'
        res=r.get('result') or {}
        urls=[]
        if isinstance(res.get('output'), list):
            urls.extend([u for u in res['output'] if isinstance(u, str)])
        if isinstance(res.get('assets'), list):
            urls.extend([a.get('url') for a in res['assets'] if isinstance(a, dict) and a.get('url')])
        if not urls: continue
        for i,u in enumerate(urls):
            dest=OUTDIR/(f"{rid}_{i}.mp4")
            try:
                download(u, dest)
                n+=1
            except Exception as e:
                print(f'warn: {e}')
    print(json.dumps({'downloaded': n, 'out_dir': str(OUTDIR)}, indent=2))

if __name__=='__main__':
    main()

