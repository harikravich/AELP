#!/usr/bin/env python3
from __future__ import annotations
import os, time, csv, io, json
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

def run_report(report_id: str, params: Dict[str, Any]) -> str:
    """Try the official Run endpoint first; if forbidden, fall back to ReportExport.

    Returns a URI that can be polled for download ("/Jobs/<id>/Download").
    """
    mode, sid, tok = _auth()
    headers={'Accept':'application/json','Content-Type':'application/json'}

    # 1) Try Reports/{id}/Run
    url=f"{API_BASE}/Advertisers/{sid}/Reports/{report_id}/Run"
    if mode=='basic':
        r=requests.post(url, auth=(sid,tok), headers=headers, json=params, timeout=60)
    else:
        r=requests.post(url, headers={'Authorization':f'Bearer {tok}', **headers}, json=params, timeout=60)
    if r.status_code in (200,201):
        data=r.json()
        run_uri=data.get('Uri') or data.get('uri') or data.get('RunUri') or data.get('run_uri')
        if not run_uri:
            run_id=data.get('Id') or data.get('id')
            if not run_id:
                raise SystemExit(f"No run id in response: {data}")
            run_uri=f"/Advertisers/{sid}/ReportRuns/{run_id}"
        # For Run API, /ReportRuns/<id>.csv is the download handle
        return run_uri

    # 2) Fallback: ReportExport/{id}.json (async job → Jobs/<id>/Download)
    if r.status_code in (401,403):
        # Build querystring; the export endpoint expects flat params, not JSON body
        export_url=f"{API_BASE}/Advertisers/{sid}/ReportExport/{report_id}.json"
        # Map our params dict to flat querystring
        # Impact accepts both START_DATE/END_DATE and Show/Event Type, etc.
        if mode=='basic':
            er = requests.get(export_url, auth=(sid,tok), params=params, headers={'Accept':'application/json'}, timeout=60)
        else:
            er = requests.get(export_url, headers={'Authorization':f'Bearer {tok}','Accept':'application/json'}, params=params, timeout=60)
        if er.status_code != 200:
            raise SystemExit(f"ReportExport error {er.status_code}: {er.text}")
        data=er.json()
        # Example keys: Status, QueuedUri, ResultUri
        # We return the ResultUri base so caller can append .csv/.json for download
        result_uri = data.get('ResultUri') or data.get('result_uri')
        if not result_uri:
            raise SystemExit(f"Unexpected ReportExport response: {data}")
        return result_uri

    raise SystemExit(f"Run error {r.status_code}: {r.text}")

def poll_csv(run_or_result_uri: str, wait_s: int=2, max_tries: int=90) -> str:
    """Polls a Run or Jobs/Result URI until CSV is ready. Follows redirects to GCS.

    Accepts either:
      - /ReportRuns/<id> (Run API)
      - /Jobs/<id>/Download (ReportExport API)
    """
    mode, sid, tok=_auth()
    # Normalize to CSV URL
    if run_or_result_uri.endswith('.csv') or run_or_result_uri.endswith('.json'):
        base = run_or_result_uri.rsplit('.',1)[0]
    else:
        base = run_or_result_uri
    csv_url=f"{API_BASE}{base}.csv"
    headers={'Accept':'text/csv'}
    for i in range(max_tries):
        if mode=='basic':
            r=requests.get(csv_url, auth=(sid,tok), headers=headers, timeout=60, allow_redirects=True)
        else:
            r=requests.get(csv_url, headers={'Authorization':f'Bearer {tok}', **headers}, timeout=60, allow_redirects=True)
        if r.status_code==200 and r.text.strip():
            return r.text
        time.sleep(wait_s)
    raise SystemExit(f"CSV still empty after {max_tries} tries: last status {r.status_code}")

def normalize_rows(csv_text: str) -> List[Dict[str, Any]]:
    reader=csv.DictReader(io.StringIO(csv_text))
    out=[]
    for row in reader:
        cols={k.strip().lower(): (v.strip() if isinstance(v,str) else v) for k,v in row.items()}
        def pick(*names, default=None):
            for n in names:
                v=cols.get(n)
                if v not in (None,''):
                    return v
            return default
        date=pick('date','date_','day','date_start')
        partner_id=pick('media partner id','media_partner_id','partner id','partner_id')
        partner=pick('media partner','media_partner','partner','partner name')
        actions=pick('actions','sales','leads','convertedcalls', default='0')
        payout=pick('totalcost','actionscost','salecost','leadcost','clientcost','cpccost', default='0')
        revenue=pick('revenue','salerevenue','callrevenue','mobileinstallrevenue','datapostrevenue', default='0')
        def to_f(x):
            try: return float(x)
            except: return 0.0
        if not date:
            continue
        out.append({'date': date, 'partner_id': partner_id or None, 'partner': partner or None,
                    'actions': to_f(actions), 'payout': to_f(payout), 'revenue': to_f(revenue)})
    return out

def load_bq(rows: List[Dict[str,Any]]):
    project=os.getenv('GOOGLE_CLOUD_PROJECT'); dataset=os.getenv('BIGQUERY_TRAINING_DATASET')
    if not project or not dataset:
        raise SystemExit('Set GOOGLE_CLOUD_PROJECT and BIGQUERY_TRAINING_DATASET')
    bq=bigquery.Client(project=project)
    table_id=f"{project}.{dataset}.impact_partner_performance"
    try: bq.get_table(table_id)
    except: 
        schema=[bigquery.SchemaField('date','DATE','REQUIRED'),
                bigquery.SchemaField('partner_id','STRING'),
                bigquery.SchemaField('partner','STRING'),
                bigquery.SchemaField('actions','FLOAT'),
                bigquery.SchemaField('payout','FLOAT'),
                bigquery.SchemaField('revenue','FLOAT')]
        t=bigquery.Table(table_id, schema=schema)
        t.time_partitioning=bigquery.TimePartitioning(field='date')
        bq.create_table(t)
    # delete range first
    if rows:
        s=min(r['date'] for r in rows)
        e=max(r['date'] for r in rows)
        bq.query(f"DELETE FROM `{table_id}` WHERE date BETWEEN DATE('{s}') AND DATE('{e}')").result()
    bq.insert_rows_json(table_id, rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--report', required=True)
    ap.add_argument('--start', required=True)
    ap.add_argument('--end', required=True)
    ap.add_argument('--event', default=None)
    args=ap.parse_args()
    params={
        'START_DATE': args.start,
        'END_DATE': args.end,
        'Show': 'All'
    }
    if args.event:
        params['Event Type']=args.event
    # SUBAID (program) if provided
    subaid=os.getenv('IMPACT_SUBAID')
    if subaid:
        params['SUBAID']=subaid
    # Some exports need explicit SHOW_* flags; pass-through any extras from env
    extras=os.getenv('IMPACT_EXTRA_PARAMS')
    if extras:
        for chunk in extras.replace('&',',').split(','):
            if '=' in chunk:
                k,v=chunk.split('=',1); params[k.strip()]=v.strip()

    run_or_result_uri=run_report(args.report, params)
    csv_text=poll_csv(run_or_result_uri)
    rows=normalize_rows(csv_text)
    load_bq(rows)
    print(f"Loaded {len(rows)} rows from run {run_uri}")

if __name__=='__main__':
    main()
