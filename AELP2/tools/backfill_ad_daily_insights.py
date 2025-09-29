#!/usr/bin/env python3
from __future__ import annotations
"""
Backfill ad-level daily insights in small date chunks with checkpointing.
Writes JSONL shards per chunk under AELP2/raw/ad_daily/YYYYMMDD_YYYYMMDD.jsonl
Fields: ad_id, adset_id, campaign_id, date_start, impressions, clicks, spend, frequency, actions.
"""
import os, json, time
from datetime import date, timedelta
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / 'AELP2' / 'raw' / 'ad_daily'
OUTDIR.mkdir(parents=True, exist_ok=True)
META_BASE = os.getenv('META_BASE_URL','https://graph.facebook.com/v23.0')

def read_env():
    tok=os.getenv('META_ACCESS_TOKEN') or os.getenv('META_ACCESS_TOKEN_DISABLED')
    acct=os.getenv('META_ACCOUNT_ID')
    if not (tok and acct) and (ROOT/'.env').exists():
        for ln in (ROOT/'.env').read_text().splitlines():
            if ln.startswith('export META_ACCESS_TOKEN=') and not tok: tok=ln.split('=',1)[1].strip()
            if ln.startswith('export META_ACCESS_TOKEN_DISABLED=') and not tok: tok=ln.split('=',1)[1].strip()
            if ln.startswith('export META_ACCOUNT_ID=') and not acct: acct=ln.split('=',1)[1].strip()
    if not (tok and acct): raise RuntimeError('Missing META creds')
    return tok, acct

def fetch_chunk(tok: str, acct: str, since: str, until: str):
    url=f"{META_BASE}/{acct}/insights"
    params={
        'time_increment': 1,
        'level': 'ad',
        'fields': 'ad_id,ad_name,adset_id,campaign_id,date_start,impressions,clicks,spend,frequency,actions',
        'time_range': json.dumps({'since': since, 'until': until}),
        'access_token': tok,
        'limit': 500
    }
    out=[]; pages=0; u=url
    while True:
        r=requests.get(u, params=params if pages==0 else None, timeout=120)
        if r.status_code>=400:
            # simple backoff and retry once without params
            time.sleep(1.0)
            rr=requests.get(u, params=None, timeout=120)
            if rr.status_code>=400:
                raise RuntimeError(f"HTTP {rr.status_code} for chunk {since}-{until}")
            r=rr
        js=r.json(); out.extend(js.get('data',[]))
        nxt=(js.get('paging') or {}).get('next')
        if not nxt: break
        u=nxt; pages+=1
        if pages>5000: break
    return out

def daterange_chunks(start: date, end: date, step_days: int=14):
    cur=start
    while cur<=end:
        nxt=min(end, cur+timedelta(days=step_days-1))
        yield cur, nxt
        cur = nxt + timedelta(days=1)

def main():
    tok, acct = read_env()
    days=int(os.getenv('AELP2_BACKFILL_DAYS','365'))
    step=int(os.getenv('AELP2_BACKFILL_CHUNK','14'))
    until=date.today()-timedelta(days=1)
    since=until - timedelta(days=days-1)
    for s,e in daterange_chunks(since, until, step):
        shard=OUTDIR / f"{s.strftime('%Y%m%d')}_{e.strftime('%Y%m%d')}.jsonl"
        if shard.exists():
            print(f'skip {shard.name}'); continue
        try:
            rows=fetch_chunk(tok, acct, s.isoformat(), e.isoformat())
            with shard.open('w', encoding='utf-8') as f:
                for r in rows:
                    f.write(json.dumps(r)+'\n')
            print(f'wrote {shard.name} ({len(rows)})')
            time.sleep(0.2)
        except Exception as ex:
            print('error', s, e, ex)

if __name__=='__main__':
    main()

