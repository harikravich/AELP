#!/usr/bin/env python3
from __future__ import annotations
"""
Lightweight crawler to fetch video titles/descriptions from public YouTube channel pages without API keys.
Provide comma-separated channel URLs in YT_CHANNELS env, e.g.:
  export YT_CHANNELS="https://www.youtube.com/@Aura,https://www.youtube.com/@AuraPrivacy"
Writes AELP2/reports/youtube_copy.json
"""
import os, re, json
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'AELP2' / 'reports' / 'youtube_copy.json'

def extract_initial_data(html: str):
    m = re.search(r"ytInitialData\"\]\s*=\s*(\{.*?\});", html, re.DOTALL)
    if not m:
        m = re.search(r"var ytInitialData\s*=\s*(\{.*?\});", html, re.DOTALL)
    if not m:
        return None
    try:
        js = json.loads(m.group(1))
        return js
    except Exception:
        return None

def collect_titles(js):
    titles=[]
    def walk(x):
        if isinstance(x, dict):
            if 'title' in x and isinstance(x['title'], dict) and 'simpleText' in x['title']:
                titles.append(x['title']['simpleText'])
            for v in x.values(): walk(v)
        elif isinstance(x, list):
            for v in x: walk(v)
    walk(js)
    # Dedup and filter short
    out=[]; seen=set()
    for t in titles:
        t=t.strip()
        if len(t)<6: continue
        if t not in seen:
            out.append(t); seen.add(t)
    return out[:200]

def main():
    channels=os.getenv('YT_CHANNELS','https://www.youtube.com/@Aura').split(',')
    agg=[]
    tried=[]
    for url in [u.strip() for u in channels if u.strip()]:
        tried.append(url)
        try:
            r=requests.get(url, timeout=20, headers={'User-Agent':'Mozilla/5.0'})
            if r.status_code!=200: continue
            js=extract_initial_data(r.text)
            if not js: continue
            titles=collect_titles(js)
            for t in titles:
                agg.append({'channel': url, 'title': t})
        except Exception:
            continue
    OUT.write_text(json.dumps({'tried': tried, 'count': len(agg), 'items': agg[:500]}, indent=2))
    print(json.dumps({'count': len(agg)}, indent=2))

if __name__=='__main__':
    main()

