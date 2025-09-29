#!/usr/bin/env python3
from __future__ import annotations
"""
Pull affiliate (Impact) copy-like fields from BigQuery tables populated by AELP2 pipelines.

Tables considered (if exist):
- impact_ads (entity dump)
- impact_media_partners (entity dump)
- impact_partner_performance (report rows; name fields only)

Writes AELP2/reports/impact_copy.json with items [{field,text}].
"""
import os, json
from pathlib import Path
from google.cloud import bigquery

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'AELP2' / 'reports' / 'impact_copy.json'

def table_exists(bq, table: str) -> bool:
    try:
        bq.get_table(table)
        return True
    except Exception:
        return False

def main():
    project = os.getenv('GOOGLE_CLOUD_PROJECT'); dataset = os.getenv('BIGQUERY_TRAINING_DATASET')
    if not project or not dataset:
        OUT.write_text(json.dumps({'error':'missing GCP env'}, indent=2)); print('{"status":"error"}'); return
    ds = f"{project}.{dataset}"
    bq = bigquery.Client(project=project)
    items = []
    # impact_ads: try text-ish fields
    t_ads = f"{ds}.impact_ads"
    if table_exists(bq, t_ads):
        q = f"""
        SELECT DISTINCT
          SAFE_CAST(JSON_VALUE(cast(t AS STRING),'$.Headline') AS STRING) AS headline,
          SAFE_CAST(JSON_VALUE(cast(t AS STRING),'$.Description') AS STRING) AS description,
          SAFE_CAST(JSON_VALUE(cast(t AS STRING),'$.Name') AS STRING) AS name
        FROM `{t_ads}`, UNNEST([TO_JSON_STRING(impact_ads)]) AS t
        LIMIT 5000
        """
        try:
            rows = list(bq.query(q).result())
            for r in rows:
                if r.headline: items.append({'field':'Headline','text': r.headline})
                if r.description: items.append({'field':'Description','text': r.description})
                if r.name: items.append({'field':'Name','text': r.name})
        except Exception:
            pass
    # impact_media_partners: Name, Website
    t_mp = f"{ds}.impact_media_partners"
    if table_exists(bq, t_mp):
        q = f"SELECT DISTINCT Name, Website FROM `{t_mp}` LIMIT 5000"
        for r in bq.query(q).result():
            if r.Name: items.append({'field':'PartnerName','text': r.Name})
            if r.Website: items.append({'field':'PartnerWebsite','text': r.Website})
    # impact_partner_performance: Partner
    t_perf = f"{ds}.impact_partner_performance"
    if table_exists(bq, t_perf):
        q = f"SELECT DISTINCT partner FROM `{t_perf}` WHERE partner IS NOT NULL LIMIT 5000"
        for r in bq.query(q).result():
            items.append({'field':'Partner','text': r.partner})
    # Write
    OUT.write_text(json.dumps({'count': len(items), 'items': items[:200]}, indent=2))
    print(json.dumps({'count': len(items)}, indent=2))

if __name__=='__main__':
    main()

