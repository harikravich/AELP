#!/usr/bin/env python3
import os
from datetime import datetime
from google.cloud import bigquery

def main():
    project=os.environ['GOOGLE_CLOUD_PROJECT']
    dataset=os.environ['BIGQUERY_TRAINING_DATASET']
    bq=bigquery.Client(project=project)
    tbl=f"{project}.{dataset}.ops_alerts"
    schema=[
        bigquery.SchemaField('timestamp','TIMESTAMP','REQUIRED'),
        bigquery.SchemaField('alert_date','DATE'),
        bigquery.SchemaField('level','STRING'),
        bigquery.SchemaField('type','STRING'),
        bigquery.SchemaField('campaign_id','STRING'),
        bigquery.SchemaField('message','STRING'),
    ]
    # Recreate table to keep schema current
    try:
        bq.delete_table(tbl, not_found_ok=True)
    except Exception:
        pass
    t=bigquery.Table(tbl, schema=schema)
    bq.create_table(t)

    q=f"""
    SELECT DATE(date) AS d, campaign_id, ANY_VALUE(campaign_name) AS name,
           AVG(impression_share) AS is_avg, AVG(lost_is_rank) AS lir,
           SAFE_DIVIDE(SUM(clicks), NULLIF(SUM(impressions),0)) AS ctr
    FROM `{project}.{dataset}.ads_campaign_performance`
    WHERE DATE(date)=DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY)
    GROUP BY d, campaign_id
    """
    now=datetime.utcnow().isoformat()
    rows=list(bq.query(q).result())
    alerts=[]
    for r in rows:
        name=(r['name'] or '').lower()
        brand=('brand' in name) or ('aura' in name)
        if r['lir'] is not None and r['lir']>0.7:
            alerts.append({'timestamp': now, 'alert_date': r['d'].isoformat(), 'level':'high','type':'rank_limited','campaign_id':r['campaign_id'], 'message':f"lost_is_rank={r['lir']:.2f}, IS={r['is_avg']}"})
        if brand and r['is_avg'] is not None and r['is_avg']<0.8:
            alerts.append({'timestamp': now, 'alert_date': r['d'].isoformat(), 'level':'high','type':'brand_is_low','campaign_id':r['campaign_id'], 'message':f"brand IS={r['is_avg']}"})
        if not brand and r['is_avg'] is not None and r['is_avg']<0.2:
            alerts.append({'timestamp': now, 'alert_date': r['d'].isoformat(), 'level':'med','type':'is_low','campaign_id':r['campaign_id'], 'message':f"IS={r['is_avg']}"})
        if r['ctr']<0.02:
            alerts.append({'timestamp': now, 'alert_date': r['d'].isoformat(), 'level':'med','type':'ctr_low','campaign_id':r['campaign_id'], 'message':f"CTR={r['ctr']:.3%}"})
    if alerts:
        bq.load_table_from_json(alerts, tbl, job_config=bigquery.LoadJobConfig(write_disposition='WRITE_APPEND')).result()
    print(f"Wrote {len(alerts)} alerts to {tbl}")

if __name__=='__main__':
    main()
