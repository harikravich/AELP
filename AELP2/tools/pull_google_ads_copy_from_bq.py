#!/usr/bin/env python3
from __future__ import annotations
"""
Pull Google Ads ad names/headlines from BigQuery (preferred over API on this box).

Tables used (if present):
- `<project>.<dataset>.ads_ad_performance` → uses `ad_name` when not redacted
- `<project>.<dataset>.ads_assets` → uses `text` for asset records (if ingested)

Writes AELP2/reports/google_ads_copy.json with fields: {headlines:[], descriptions:[]}
Note: Google Ads RSA descriptions are not typically in these tables unless assets ingestion is run.
"""
import os, json
from pathlib import Path
from google.cloud import bigquery

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'AELP2' / 'reports' / 'google_ads_copy.json'

def table_exists(bq: bigquery.Client, table: str) -> bool:
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
    headlines = []
    descriptions = []
    # Prefer ad_name from ads_ad_performance where not null
    t1 = f"{ds}.ads_ad_performance"
    if table_exists(bq, t1):
        q = f"""
        SELECT DISTINCT ad_name
        FROM `{t1}`
        WHERE ad_name IS NOT NULL AND ad_name != ''
        ORDER BY ad_name
        LIMIT 1000
        """
        rows = list(bq.query(q).result())
        headlines.extend([r.ad_name for r in rows if r.ad_name])
    # Also collect any text assets if ads_assets exists
    t2 = f"{ds}.ads_assets"
    if table_exists(bq, t2):
        q2 = f"""
        SELECT DISTINCT text
        FROM `{t2}`
        WHERE text IS NOT NULL AND text != ''
        ORDER BY text
        LIMIT 2000
        """
        rows = list(bq.query(q2).result())
        # Assets text could be either headlines or descriptions; we bucket by length
        for r in rows:
            txt = r.text.strip()
            if len(txt) <= 30:
                headlines.append(txt)
            else:
                descriptions.append(txt)
    out = {
        'headlines': [{'text': h, 'count': 1} for h in sorted(set(headlines))][:500],
        'descriptions': [{'text': d, 'count': 1} for d in sorted(set(descriptions))][:500]
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps({'headlines': len(out['headlines']), 'descriptions': len(out['descriptions'])}, indent=2))

if __name__=='__main__':
    main()

