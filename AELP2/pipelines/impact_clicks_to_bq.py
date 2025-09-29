#!/usr/bin/env python3
from __future__ import annotations
import os, time, csv, io
import argparse
import requests
from typing import Dict, Any, List
from google.cloud import bigquery

API_BASE = "https://api.impact.com"

def _auth():
    sid=os.getenv('IMPACT_ACCOUNT_SID'); bearer=os.getenv('IMPACT_BEARER_TOKEN'); basic=os.getenv('IMPACT_AUTH_TOKEN')
    if sid and basic:
        return 'basic', sid, basic
    if bearer and sid:
        return 'bearer', sid, bearer
    raise SystemExit('Set IMPACT_ACCOUNT_SID and either IMPACT_AUTH_TOKEN or IMPACT_BEARER_TOKEN')

def start_export(report_id: str, params: Dict[str, Any]) -> Dict[str,str]:
    mode, sid, tok=_auth()
    url=f"{API_BASE}/Advertisers/{sid}/ReportExport/{report_id}.json"
    headers={'Accept':'application/json'}
    if mode=='basic':
        r=requests.get(url, auth=(sid,tok), params=params, headers=headers, timeout=60)
    else:
        r=requests.get(url, headers={'Authorization':f'Bearer {tok}', **headers}, params=params, timeout=60)
    if r.status_code!=200:
        raise SystemExit(f"Export start failed {r.status_code}: {r.text}")
    data=r.json()
    queued=data.get('QueuedUri') or data.get('Queued') or data.get('queued_uri')
    res=data.get('ResultUri') or data.get('result_uri')
    return {'queued_uri': queued, 'result_uri': res}

def download_csv(job_uris: Dict[str,str], timeout_s: int=600) -> str:
    mode, sid, tok=_auth()
    headers={'Accept':'application/json'}
    # Poll job status if queued
    if job_uris.get('queued_uri'):
        status_url=f"{API_BASE}{job_uris['queued_uri']}"
        deadline=time.time()+timeout_s
        while time.time() < deadline:
            if mode=='basic':
                s=requests.get(status_url, auth=(sid,tok), headers=headers, timeout=60)
            else:
                s=requests.get(status_url, headers={'Authorization':f'Bearer {tok}', **headers}, timeout=60)
            if s.status_code==200:
                j=s.json();
                if (j.get('Status') or '').upper()== 'COMPLETED':
                    break
            time.sleep(2)
    # Download result (CSV), follow redirects
    result_uri = job_uris.get('result_uri') or (requests.get(status_url).json().get('ResultUri') if 'status_url' in locals() else None)
    if not result_uri:
        raise SystemExit('Missing ResultUri for click export')
    csv_url=f"{API_BASE}{result_uri}.csv"
    headers={'Accept':'text/csv'}
    for _ in range(int(timeout_s/2)):
        if mode=='basic':
            r=requests.get(csv_url, auth=(sid,tok), headers=headers, timeout=60, allow_redirects=True)
        else:
            r=requests.get(csv_url, headers={'Authorization':f'Bearer {tok}', **headers}, timeout=60, allow_redirects=True)
        if r.status_code==200 and r.text.strip():
            return r.text
        time.sleep(2)
    raise SystemExit('Timed out waiting for click CSV')

def normalize_clicks(csv_text: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    r=csv.DictReader(io.StringIO(csv_text))
    for row in r:
        g=lambda k: (row.get(k) or '').strip()
        date = g('Date') or g('date') or g('date_display') or g('date_')
        click_id = g('Click Id') or g('click_id') or g('ClickId')
        ad_id = g('Ad Id') or g('ad_id') or g('AdId')
        pid = g('Media Partner Id') or g('Media Partner ID') or g('media_id') or g('Partner Id')
        partner = g('Media Partner') or g('media_partner') or g('Partner')
        ts = g('Click Date Time') or g('ClickDateTime') or ''
        if not (date and click_id):
            continue
        out.append({'date': date[:10], 'ts': ts, 'click_id': click_id, 'ad_id': ad_id or None,
                    'partner_id': pid or None, 'partner': partner or None})
    return out

def ensure_table(bq: bigquery.Client, project: str, dataset: str) -> str:
    tid=f"{project}.{dataset}.impact_clicks"
    try:
        bq.get_table(tid)
        return tid
    except Exception:
        schema=[
            bigquery.SchemaField('date','DATE','REQUIRED'),
            bigquery.SchemaField('ts','STRING'),
            bigquery.SchemaField('click_id','STRING','REQUIRED'),
            bigquery.SchemaField('ad_id','STRING'),
            bigquery.SchemaField('partner_id','STRING'),
            bigquery.SchemaField('partner','STRING'),
        ]
        t=bigquery.Table(tid, schema=schema)
        t.time_partitioning=bigquery.TimePartitioning(field='date')
        bq.create_table(t)
        return tid

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--start', required=True)
    ap.add_argument('--end', required=True)
    args=ap.parse_args()
    project=os.getenv('GOOGLE_CLOUD_PROJECT'); dataset=os.getenv('BIGQUERY_TRAINING_DATASET')
    if not project or not dataset:
        raise SystemExit('Set GOOGLE_CLOUD_PROJECT and BIGQUERY_TRAINING_DATASET')
    params={'START_DATE': args.start, 'END_DATE': args.end, 'Show':'All', 'PageSize': 200000}
    job = start_export('payable_click_data', params)
    csv_text = download_csv(job)
    rows = normalize_clicks(csv_text)
    bq=bigquery.Client(project=project)
    tid=ensure_table(bq, project, dataset)
    if rows:
        s=min(r['date'] for r in rows); e=max(r['date'] for r in rows)
        bq.query(f"DELETE FROM `{tid}` WHERE date BETWEEN DATE('{s}') AND DATE('{e}')").result()
        # chunk insert
        for i in range(0, len(rows), 10000):
            batch=rows[i:i+10000]
            errs=bq.insert_rows_json(tid, batch)
            if errs:
                raise SystemExit(f"BQ insert errors: {errs[:3]}")
    print(f"Loaded {len(rows)} clicks into {tid}")

if __name__=='__main__':
    main()
