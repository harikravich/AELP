#!/usr/bin/env python3
from __future__ import annotations
"""
Pull top affiliate (Impact) copy snippets via Impact API.
Writes AELP2/reports/impact_copy.json

Env:
- IMPACT_ACCOUNT_SID
- IMPACT_AUTH_TOKEN (basic auth)

Note: Endpoint paths differ by role; this script tries a couple of common advertiser endpoints and records the first success.
"""
import os, json, time
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'AELP2' / 'reports' / 'impact_copy.json'

def main():
    sid=os.getenv('IMPACT_ACCOUNT_SID'); tok=os.getenv('IMPACT_AUTH_TOKEN')
    if not (sid and tok):
        OUT.write_text(json.dumps({'error':'missing creds'}, indent=2)); print('{"status":"error","detail":"missing creds"}'); return
    sess=requests.Session(); sess.auth=(sid, tok)
    tried=[]; copies=[]
    # Prefer v13 Advertiser endpoints known to work with current creds
    endpoints=[
        f'https://api.impact.com/Advertisers/{sid}/Campaigns',
        f'https://api.impact.com/Advertisers/{sid}/Reports'
    ]
    for url in endpoints:
        tried.append(url)
        try:
            r=sess.get(url, timeout=20, headers={'Accept':'application/json'})
            if r.status_code==200:
                js=r.json()
                # Try to extract headings/names/descriptions if present
                def walk(x):
                    if isinstance(x, dict):
                        for k,v in x.items():
                            key=k.lower()
                            if key in ('name','headline','title','description','body','text','caption') and isinstance(v, str):
                                copies.append({'field': k, 'text': v})
                            walk(v)
                    elif isinstance(x, list):
                        for it in x: walk(it)
                walk(js)
                break
        except Exception as e:
            continue
    OUT.write_text(json.dumps({'tried': tried, 'count': len(copies), 'items': copies[:200]}, indent=2))
    print(json.dumps({'count': len(copies)}, indent=2))

if __name__=='__main__':
    main()
