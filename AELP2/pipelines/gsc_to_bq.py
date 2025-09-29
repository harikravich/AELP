#!/usr/bin/env python3
"""
Google Search Console → BigQuery (brand vs non-brand trend).
Env:
  GSC_SITE_URL, GOOGLE_CLOUD_PROJECT, BIGQUERY_TRAINING_DATASET
  For auth: Application Default Credentials with webmasters.readonly scope.
Writes:
  <project>.<dataset>.gsc_brand_monthly (month, brand_clicks, all_clicks, brand_share)
"""
import os
from datetime import date, timedelta
from google.cloud import bigquery

def main():
    site=os.environ.get('GSC_SITE_URL')
    project=os.environ['GOOGLE_CLOUD_PROJECT']
    dataset=os.environ['BIGQUERY_TRAINING_DATASET']
    if not site:
        print('GSC_SITE_URL not set; skipping')
        return
    try:
        from googleapiclient.discovery import build
    except Exception as e:
        print(f'googleapiclient missing: {e}; skipping')
        return
    service=build('searchconsole','v1') if False else build('webmasters','v3')
    end=date.today()
    start=end - timedelta(days=365)
    request={
        'startDate': start.isoformat(),
        'endDate': end.isoformat(),
        'dimensions':['date','query'],
        'rowLimit': 25000,
    }
    # Best-effort: GSC API shape may differ; handle failures gracefully
    try:
        resp=service.searchanalytics().query(siteUrl=site, body=request).execute()
    except Exception as e:
        print(f'GSC query failed: {e}; skipping')
        return
    rows=resp.get('rows',[])
    # Classify brand by query containing 'aura' (tweak later with regex config)
    monthly={}
    for r in rows:
        keys=r.get('keys',[])
        if len(keys)<2: continue
        d=keys[0]; q=keys[1].lower()
        y,m = d.split('-')[0], d.split('-')[1]
        month=f"{y}-{m}-01"
        clicks=float(r.get('clicks',0))
        rec=monthly.setdefault(month, {'brand':0.0,'all':0.0})
        rec['all']+=clicks
        if 'aura' in q or 'brand' in q:
            rec['brand']+=clicks
    out=[{'month': k, 'brand_clicks': v['brand'], 'all_clicks': v['all'], 'brand_share': (v['brand']/v['all'] if v['all'] else None)} for k,v in monthly.items()]
    if not out:
        print('No GSC rows aggregated; skipping')
        return
    bq=bigquery.Client(project=project)
    table=f"{project}.{dataset}.gsc_brand_monthly"
    schema=[
        bigquery.SchemaField('month','DATE','REQUIRED'),
        bigquery.SchemaField('brand_clicks','FLOAT'),
        bigquery.SchemaField('all_clicks','FLOAT'),
        bigquery.SchemaField('brand_share','FLOAT'),
    ]
    try:
        bq.delete_table(table, not_found_ok=True)
    except Exception: pass
    t=bigquery.Table(table, schema=schema)
    bq.create_table(t)
    for r in out:
        r['month']=r['month']
    bq.load_table_from_json(out, table, job_config=bigquery.LoadJobConfig(write_disposition='WRITE_TRUNCATE')).result()
    print(f'Loaded {len(out)} GSC monthly rows into {table}')

if __name__=='__main__':
    main()

