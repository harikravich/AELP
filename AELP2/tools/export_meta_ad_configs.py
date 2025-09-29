#!/usr/bin/env python3
from __future__ import annotations
"""
Export current campaign and adset configuration from Meta Graph API.
Writes JSONL snapshots under AELP2/raw/ad_config/{campaigns,adsets}.jsonl

Fields captured (best-effort):
- Campaign: id, name, objective, bid_strategy, status, configured_status, special_ad_categories, created_time, updated_time
- Adset: id, name, campaign_id, optimization_goal, bid_strategy, is_dynamic_creative, targeting,
         daily_budget, effective_status, created_time, updated_time

Auth: reads META_ACCESS_TOKEN or META_ACCESS_TOKEN_DISABLED and META_ACCOUNT_ID from env or .env
"""
import os, json, time, sys
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / 'AELP2' / 'raw' / 'ad_config'
OUTDIR.mkdir(parents=True, exist_ok=True)
BASE = os.getenv('META_BASE_URL','https://graph.facebook.com/v23.0')

def read_env():
    tok=os.getenv('META_ACCESS_TOKEN') or os.getenv('META_ACCESS_TOKEN_DISABLED')
    acct=os.getenv('META_ACCOUNT_ID')
    if not (tok and acct) and (ROOT/'.env').exists():
        for ln in (ROOT/'.env').read_text().splitlines():
            if ln.startswith('export META_ACCESS_TOKEN=') and not tok: tok=ln.split('=',1)[1].strip()
            if ln.startswith('export META_ACCESS_TOKEN_DISABLED=') and not tok: tok=ln.split('=',1)[1].strip()
            if ln.startswith('export META_ACCOUNT_ID=') and not acct: acct=ln.split('=',1)[1].strip()
    if not (tok and acct):
        raise RuntimeError('Missing META creds (META_ACCESS_TOKEN[_DISABLED], META_ACCOUNT_ID)')
    return tok, acct

def fetch_paged(url: str, params: dict):
    out=[]; pages=0; u=url; p=params
    while True:
        r=requests.get(u, params=p if pages==0 else None, timeout=120)
        if r.status_code>=400:
            # one retry without params
            time.sleep(1.0)
            rr=requests.get(u, timeout=120)
            if rr.status_code>=400:
                raise RuntimeError(f"HTTP {rr.status_code}: {r.text[:120]}")
            r=rr
        js=r.json(); out.extend(js.get('data',[]))
        nxt=(js.get('paging') or {}).get('next')
        if not nxt: break
        u=nxt; pages+=1
        if pages>10000: break
    return out

def export_campaigns(tok: str, acct: str):
    url=f"{BASE}/{acct}/campaigns"
    fields='id,name,objective,bid_strategy,status,configured_status,special_ad_categories,created_time,updated_time'
    rows=fetch_paged(url, {'fields': fields, 'access_token': tok, 'limit': 500})
    fp=OUTDIR/'campaigns.jsonl'
    with fp.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r)+'\n')
    print(f"wrote {fp} ({len(rows)})")

def export_adsets(tok: str, acct: str):
    url=f"{BASE}/{acct}/adsets"
    fields='id,name,campaign_id,optimization_goal,bid_strategy,is_dynamic_creative,targeting,daily_budget,effective_status,created_time,updated_time'
    rows=fetch_paged(url, {'fields': fields, 'access_token': tok, 'limit': 500})
    fp=OUTDIR/'adsets.jsonl'
    with fp.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r)+'\n')
    print(f"wrote {fp} ({len(rows)})")

def main():
    tok, acct = read_env()
    export_campaigns(tok, acct)
    export_adsets(tok, acct)

if __name__=='__main__':
    main()
