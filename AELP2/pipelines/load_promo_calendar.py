#!/usr/bin/env python3
import os, csv
from google.cloud import bigquery

def main():
    project=os.environ['GOOGLE_CLOUD_PROJECT']
    dataset=os.environ['BIGQUERY_TRAINING_DATASET']
    path=os.environ.get('PROMO_CALENDAR_PATH','AELP2/config/promo_calendar.csv')
    bq=bigquery.Client(project=project)
    table_id=f"{project}.{dataset}.promo_calendar"
    schema=[
        bigquery.SchemaField('date','DATE','REQUIRED'),
        bigquery.SchemaField('promo_flag','INT64'),
        bigquery.SchemaField('promo_intensity','FLOAT64'),
        bigquery.SchemaField('label','STRING'),
    ]
    try:
        bq.delete_table(table_id, not_found_ok=True)
    except Exception:
        pass
    t=bigquery.Table(table_id,schema=schema)
    t.time_partitioning=bigquery.TimePartitioning(field='date')
    bq.create_table(t)
    rows=[]
    with open(path,'r') as f:
        rdr=csv.DictReader(f)
        for r in rdr:
            rows.append({
                'date': r['date'],
                'promo_flag': int(r['promo_flag'] or 0),
                'promo_intensity': float(r['promo_intensity'] or 0.0),
                'label': r.get('label','')
            })
    if rows:
        bq.load_table_from_json(rows, table_id, job_config=bigquery.LoadJobConfig(write_disposition='WRITE_TRUNCATE')).result()
    print(f"Loaded {len(rows)} promo rows into {table_id}")

if __name__=='__main__':
    main()

