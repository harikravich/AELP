#!/usr/bin/env python3
from __future__ import annotations
import os, csv
from google.cloud import bigquery

def main():
    project=os.getenv('GOOGLE_CLOUD_PROJECT','aura-thrive-platform')
    dataset=os.getenv('BIGQUERY_TRAINING_DATASET','gaelp_training')
    client=bigquery.Client(project=project)
    rows=list(client.query(f"SELECT table_name FROM `{project}.{dataset}.INFORMATION_SCHEMA.TABLES` ORDER BY table_name").result())
    out='AELP2/reports/bq_inventory.csv'
    with open(out,'w',newline='') as f:
        w=csv.writer(f)
        w.writerow(['table','row_count','latest_date'])
        for r in rows:
            tbl=f"{project}.{dataset}.{r['table_name']}"
            try:
                n=list(client.query(f"SELECT COUNT(1) n FROM `{tbl}`").result())[0]['n']
            except Exception:
                n=''
            latest=''
            for cand in ('date','ts','timestamp','_PARTITIONTIME','_PARTITIONDATE'):
                try:
                    q=f"SELECT MAX({cand}) mx FROM `{tbl}`"
                    mx=list(client.query(q).result())
                    if mx and 'mx' in mx[0] and mx[0]['mx'] is not None:
                        latest=str(mx[0]['mx']); break
                except Exception:
                    pass
            w.writerow([tbl, n, latest])
    print(out)

if __name__=='__main__':
    main()

