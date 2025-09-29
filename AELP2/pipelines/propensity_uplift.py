#!/usr/bin/env python3
"""
Propensity/Uplift Bootstrap: writes `segment_scores_daily` using simple exposed vs unexposed deltas.

If journey tables are empty, creates the table and exits gracefully.
"""
import os
import json
from google.cloud import bigquery
from google.cloud.exceptions import NotFound


def ensure_table(bq: bigquery.Client, project: str, dataset: str):
    table_id = f"{project}.{dataset}.segment_scores_daily"
    try:
        bq.get_table(table_id)
        return table_id
    except NotFound:
        schema = [
            bigquery.SchemaField('date', 'DATE', mode='REQUIRED'),
            bigquery.SchemaField('segment', 'STRING', mode='REQUIRED'),
            bigquery.SchemaField('score', 'FLOAT', mode='REQUIRED'),
            bigquery.SchemaField('method', 'STRING', mode='REQUIRED'),
            bigquery.SchemaField('notes', 'STRING', mode='NULLABLE'),
            bigquery.SchemaField('metadata', 'JSON', mode='NULLABLE'),
        ]
        t = bigquery.Table(table_id, schema=schema)
        t.time_partitioning = bigquery.TimePartitioning(type_=bigquery.TimePartitioningType.DAY, field='date')
        bq.create_table(t)
        return table_id


def main():
    project = os.getenv('GOOGLE_CLOUD_PROJECT')
    dataset = os.getenv('BIGQUERY_TRAINING_DATASET')
    if not project or not dataset:
        raise RuntimeError('Set GOOGLE_CLOUD_PROJECT and BIGQUERY_TRAINING_DATASET')
    bq = bigquery.Client(project=project)
    table_id = ensure_table(bq, project, dataset)

    # Check journey tables
    users_ds = f"{project}.gaelp_users"
    try:
        bq.get_table(f"{users_ds}.journey_sessions")
        bq.get_table(f"{users_ds}.persistent_touchpoints")
    except NotFound:
        print('Journey tables missing; created segment_scores_daily only')
        return

    # Simple proxy: reuse uplift_eval exposed vs unexposed per segment
    sql = f"""
      WITH es AS (
        SELECT segment, exposed, conversion_rate
        FROM `{project}.{dataset}.uplift_segment_daily`
        WHERE date = CURRENT_DATE()
      )
      SELECT CURRENT_DATE() AS date, segment,
             MAX(CASE WHEN exposed THEN conversion_rate ELSE NULL END) AS exposed_cr,
             MAX(CASE WHEN NOT exposed THEN conversion_rate ELSE NULL END) AS unexposed_cr
      FROM es
      GROUP BY segment
    """
    rows = [dict(r) for r in bq.query(sql).result()]
    out = []
    for r in rows:
        exp_cr = float(r.get('exposed_cr') or 0.0)
        unexp_cr = float(r.get('unexposed_cr') or 0.0)
        score = max(0.0, exp_cr - unexp_cr)
        # Simple normal-approx CI for difference of proportions with conservative n (unknown here)
        # If counts become available in uplift_segment_daily, include them; for now, document approximation
        se = None
        lo = None
        hi = None
        try:
            # Attempt to read counts if present
            cnt_sql = f"""
              SELECT segment,
                     MAX(CASE WHEN exposed THEN users ELSE NULL END) AS exp_n,
                     MAX(CASE WHEN NOT exposed THEN users ELSE NULL END) AS unexp_n
              FROM `{project}.{dataset}.uplift_segment_daily`
              WHERE date = CURRENT_DATE() AND segment = @seg
              GROUP BY segment
            """
            job = bq.query(cnt_sql, job_config=bigquery.QueryJobConfig(query_parameters=[bigquery.ScalarQueryParameter('seg','STRING', r['segment'])]))
            row = list(job.result())[0]
            exp_n = float(row.exp_n or 0)
            unexp_n = float(row.unexp_n or 0)
            if exp_n > 0 and unexp_n > 0:
                se = ( (exp_cr*(1-exp_cr))/max(exp_n,1) + (unexp_cr*(1-unexp_cr))/max(unexp_n,1) ) ** 0.5
                lo = score - 1.96*se
                hi = score + 1.96*se
        except Exception:
            pass
        meta = {'exp_cr': exp_cr, 'unexp_cr': unexp_cr, 'ci_lo': lo, 'ci_hi': hi, 'notes': 'counts missing -> CI approximate' if se is None else 'wald_95_ci'}
        out.append({'date': r['date'].isoformat(), 'segment': r['segment'], 'score': score, 'method': 'uplift_delta', 'notes': None, 'metadata': json.dumps(meta)})
    if out:
        bq.insert_rows_json(table_id, out)
        print(f"Wrote {len(out)} segment scores to {table_id}")
    else:
        print('No segment scores computed (likely no uplift rows today)')


if __name__ == '__main__':
    main()
