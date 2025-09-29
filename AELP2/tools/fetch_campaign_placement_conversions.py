#!/usr/bin/env python3
from __future__ import annotations
import json, os, time
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'AELP2' / 'reports' / 'placement_conversions'
OUT.mkdir(parents=True, exist_ok=True)
META_BASE = 'https://graph.facebook.com/v23.0'

def read_env():
    tok=os.getenv('META_ACCESS_TOKEN') or os.getenv('META_ACCESS_TOKEN_DISABLED'); acct=os.getenv('META_ACCOUNT_ID')
    if not (tok and acct) and (ROOT/'.env').exists():
        for ln in (ROOT/'.env').read_text().splitlines():
            if ln.startswith('export META_ACCESS_TOKEN=') and not tok: tok=ln.split('=',1)[1].strip()
            if ln.startswith('export META_ACCESS_TOKEN_DISABLED=') and not tok: tok=ln.split('=',1)[1].strip()
            if ln.startswith('export META_ACCOUNT_ID=') and not acct: acct=ln.split('=',1)[1].strip()
    if not (tok and acct): raise RuntimeError('Missing META creds')
    return tok, acct

def fetch_campaign_ids(tok: str, acct: str):
    url=f"{META_BASE}/{acct}/campaigns"; params={'fields':'id,name','limit':200,'access_token':tok}
    out=[]; pages=0
    while True:
        r=requests.get(url, params=params if pages==0 else None, timeout=60)
        if r.status_code>=400: break
        js=r.json(); out.extend([row.get('id') for row in js.get('data',[]) if row.get('id')])
        nxt=(js.get('paging') or {}).get('next')
        if not nxt: break
        url=nxt; pages+=1
        if pages>1000: break
    return out

def fetch_placement(tok: str, acct: str, camp_id: str, since: str, until: str):
    url=f"{META_BASE}/{acct}/insights"
    params={
        'level':'campaign',
        'filtering': json.dumps([{ 'field':'campaign.id','operator':'IN','value':[camp_id]}]),
        'breakdowns':'publisher_platform,platform_position,impression_device',
        'fields':'impressions,clicks,actions',
        'time_range': json.dumps({'since':since,'until':until}),
        'access_token': tok,
        'limit': 500
    }
    out={}; pages=0; url0=url
    while True:
        r=requests.get(url, params=params if pages==0 else None, timeout=90)
        if r.status_code>=400: break
        js=r.json()
        for row in js.get('data',[]):
            pp=row.get('publisher_platform') or 'unknown'; pos=row.get('platform_position') or 'unknown'; dev=row.get('impression_device') or 'unknown'
            key=f"{pp}|{pos}|{dev}"
            impr=float(row.get('impressions') or 0.0); clk=float(row.get('clicks') or 0.0)
            purch=0.0
            for a in (row.get('actions') or []):
                if a.get('action_type')=='offsite_conversion.fb_pixel_purchase':
                    try: purch=float(a.get('value') or 0.0)
                    except Exception: purch=0.0
            cur=out.get(key, {'impr':0.0,'clicks':0.0,'purch':0.0}); cur['impr']+=impr; cur['clicks']+=clk; cur['purch']+=purch; out[key]=cur
        nxt=(js.get('paging') or {}).get('next')
        if not nxt: break
        url=nxt; pages+=1
        if pages>4000: break
    return out

def main():
    tok, acct = read_env()
    from datetime import date, timedelta
    until=(date.today()-timedelta(days=1)).isoformat(); since=(date.today()-timedelta(days=120)).isoformat()
    camps=fetch_campaign_ids(tok, acct)
    for cid in camps:
        try:
            out=fetch_placement(tok, acct, cid, since, until)
            if out:
                (OUT/f'{cid}.json').write_text(json.dumps({'campaign_id': cid, 'since': since, 'until': until, 'data': out}, indent=2))
                print(f'wrote {cid}.json')
            time.sleep(0.2)
        except Exception as e:
            print('skip', cid, e)

if __name__=='__main__':
    main()

