#!/usr/bin/env python3
"""
LTV Priors Daily: computes simple 30/90-day LTV priors per segment from uplift scores.
Writes to <project>.<dataset>.ltv_priors_daily.
"""
import os, json
from datetime import date
from google.cloud import bigquery
from google.cloud.exceptions import NotFound


def ensure(bq: bigquery.Client, project: str, dataset: str) -> str:
    table=f"{project}.{dataset}.ltv_priors_daily"
    try:
        bq.get_table(table)
    except NotFound:
        schema=[
            bigquery.SchemaField('date','DATE'),
            bigquery.SchemaField('segment','STRING'),
            bigquery.SchemaField('ltv_30','FLOAT'),
            bigquery.SchemaField('ltv_90','FLOAT'),
            bigquery.SchemaField('method','STRING'),
            bigquery.SchemaField('metadata','JSON')
        ]
        t=bigquery.Table(table, schema=schema)
        t.time_partitioning=bigquery.TimePartitioning(type_=bigquery.TimePartitioningType.DAY, field='date')
        bq.create_table(t)
    return table


def main():
    project=os.getenv('GOOGLE_CLOUD_PROJECT'); dataset=os.getenv('BIGQUERY_TRAINING_DATASET')
    if not project or not dataset: raise RuntimeError('Set GOOGLE_CLOUD_PROJECT and BIGQUERY_TRAINING_DATASET')
    bq=bigquery.Client(project=project)
    table=ensure(bq,project,dataset)
    sql=f"SELECT segment, score FROM `{project}.{dataset}.segment_scores_daily` WHERE date=CURRENT_DATE() ORDER BY score DESC LIMIT 100"
    try:
        rows=[dict(r) for r in bq.query(sql).result()]
    except Exception:
        rows=[]
    out=[]
    for r in rows:
        s=float(r.get('score') or 0.0)
        ltv30=50.0 + 200.0*s
        ltv90=120.0 + 400.0*s
        out.append({'date': date.today().isoformat(),'segment': r['segment'],'ltv_30': ltv30,'ltv_90': ltv90,'method':'uplift_scaled','metadata': json.dumps({'score': s})})
    if out:
        bq.insert_rows_json(table,out)
        print(f'wrote {len(out)} ltv priors')
    else:
        print('no segment scores today')


if __name__=='__main__':
    main()

