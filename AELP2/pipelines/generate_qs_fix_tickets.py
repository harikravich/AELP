#!/usr/bin/env python3
import os, time
from datetime import datetime
from google.cloud import bigquery

def main():
    project=os.environ['GOOGLE_CLOUD_PROJECT']
    dataset=os.environ['BIGQUERY_TRAINING_DATASET']
    bq=bigquery.Client(project=project)
    table=f"{project}.{dataset}.qs_fix_tickets"
    schema=[
        bigquery.SchemaField('timestamp','TIMESTAMP','REQUIRED'),
        bigquery.SchemaField('campaign_id','STRING'),
        bigquery.SchemaField('campaign_name','STRING'),
        bigquery.SchemaField('issue','STRING'),
        bigquery.SchemaField('severity','STRING'),
        bigquery.SchemaField('evidence','STRING'),
        bigquery.SchemaField('suggestion','STRING'),
    ]
    try:
        bq.get_table(table)
    except Exception:
        t=bigquery.Table(table, schema=schema)
        bq.create_table(t)

    q=f"""
    WITH base AS (
      SELECT 
        campaign_id,
        ANY_VALUE(campaign_name) AS name,
        SUM(impressions) AS imps,
        SUM(clicks) AS clicks,
        SUM(cost_micros)/1e6 AS spend,
        SUM(conversions) AS conv,
        AVG(impression_share) AS is_avg,
        AVG(lost_is_rank) AS lost_is_rank,
        AVG(lost_is_budget) AS lost_is_budget
      FROM `{project}.{dataset}.ads_campaign_performance`
      WHERE DATE(date) BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 14 DAY) AND CURRENT_DATE()
      GROUP BY campaign_id
    )
    SELECT * FROM base WHERE spend > 3000
    """
    rows=list(bq.query(q).result())
    out=[]
    now=datetime.utcnow().isoformat()
    for r in rows:
        name=(r['name'] or '')
        ctr=(r['clicks']/(r['imps'] or 1)) if r['imps'] else 0
        brand=('brand' in name.lower()) or ('aura' in name.lower())
        if r['lost_is_rank'] is not None and r['lost_is_rank']>0.6:
            out.append({'timestamp': now, 'campaign_id': r['campaign_id'], 'campaign_name': name,
                        'issue':'rank_limited','severity':'high',
                        'evidence': f"lost_is_rank={r['lost_is_rank']:.2f}, IS={r['is_avg']}",
                        'suggestion':'Improve Ad Rank: split intents; strengthen RSAs; improve LP relevance; raise bids 5–10% on top STAGs'})
        if brand and r['is_avg'] is not None and r['is_avg']<0.8:
            out.append({'timestamp': now, 'campaign_id': r['campaign_id'], 'campaign_name': name,
                        'issue':'brand_is_low','severity':'high',
                        'evidence': f"brand IS={r['is_avg']}",
                        'suggestion':'Defend brand: exact brand ad groups, pin brand headlines, raise bids to IS ≥ 90%'})
        if not brand and r['is_avg'] is not None and r['is_avg']<0.2:
            out.append({'timestamp': now, 'campaign_id': r['campaign_id'], 'campaign_name': name,
                        'issue':'low_coverage','severity':'med',
                        'evidence': f"IS={r['is_avg']}",
                        'suggestion':'Add exact/phrase for top search terms; add negatives; raise bids modestly where CAC holds'})
        if ctr<0.02:
            out.append({'timestamp': now, 'campaign_id': r['campaign_id'], 'campaign_name': name,
                        'issue':'low_ctr','severity':'med',
                        'evidence': f"ctr={ctr:.3%}",
                        'suggestion':'RSA refresh: replace Low‑rated assets, add 12–15 headlines and 4 descriptions; DKI on safe groups'})

    if out:
        bq.load_table_from_json(out, table, job_config=bigquery.LoadJobConfig(write_disposition='WRITE_APPEND')).result()
    print(f"Wrote {len(out)} QS fix tickets to {table}")

if __name__=='__main__':
    main()

