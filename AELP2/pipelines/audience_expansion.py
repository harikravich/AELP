#!/usr/bin/env python3
"""
Audience Expansion Tooling (shadow-only).

Reads recent search terms/keywords from Ads tables (if present) and proposes
new keywords/audiences with simple heuristics. Writes to
`<project>.<dataset>.audience_expansion_candidates`.

Safe defaults: if inputs are missing, ensures the table and writes a single
stub row with a clear note.
"""
import os
from datetime import datetime
from google.cloud import bigquery
from google.cloud.exceptions import NotFound


def ensure_table(bq: bigquery.Client, project: str, dataset: str) -> str:
    table_id = f"{project}.{dataset}.audience_expansion_candidates"
    try:
        bq.get_table(table_id)
        return table_id
    except NotFound:
        schema = [
            bigquery.SchemaField('timestamp', 'TIMESTAMP', mode='REQUIRED'),
            bigquery.SchemaField('source', 'STRING', mode='REQUIRED'),
            bigquery.SchemaField('candidate', 'STRING', mode='REQUIRED'),
            bigquery.SchemaField('reason', 'STRING', mode='NULLABLE'),
            bigquery.SchemaField('score', 'FLOAT', mode='NULLABLE'),
            bigquery.SchemaField('shadow', 'BOOL', mode='REQUIRED'),
        ]
        t = bigquery.Table(table_id, schema=schema)
        t.time_partitioning = bigquery.TimePartitioning(type_=bigquery.TimePartitioningType.DAY, field='timestamp')
        bq.create_table(t)
        return table_id


def main():
    project = os.getenv('GOOGLE_CLOUD_PROJECT')
    dataset = os.getenv('BIGQUERY_TRAINING_DATASET')
    if not project or not dataset:
        raise RuntimeError('Set GOOGLE_CLOUD_PROJECT and BIGQUERY_TRAINING_DATASET')
    bq = bigquery.Client(project=project)
    table_id = ensure_table(bq, project, dataset)

    # Try search terms first
    rows = []
    try:
        sql = f"""
          SELECT LOWER(term) AS term,
                 SUM(clicks) AS clicks,
                 SAFE_DIVIDE(SUM(conversions), NULLIF(SUM(clicks),0)) AS cvr
          FROM `{project}.{dataset}.ads_search_terms`
          WHERE DATE(date) BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 28 DAY) AND CURRENT_DATE()
          GROUP BY term
          HAVING clicks >= 50 AND cvr >= 0.03
          ORDER BY cvr DESC, clicks DESC
          LIMIT 20
        """
        rows = [dict(r) for r in bq.query(sql).result()]
    except Exception:
        pass

    out = []
    now = datetime.utcnow().isoformat()
    if rows:
        for r in rows:
            out.append({
                'timestamp': now,
                'source': 'ads_search_terms',
                'candidate': r['term'],
                'reason': f"clicks={int(r['clicks'] or 0)}, cvr={float(r['cvr'] or 0.0):.3f}",
                'score': float(r['cvr'] or 0.0),
                'shadow': True,
            })
    else:
        # Fallback stub row
        out.append({
            'timestamp': now,
            'source': 'stub',
            'candidate': 'example_audience_or_keyword',
            'reason': 'ads_search_terms not available; provide candidates from BQ when present',
            'score': 0.0,
            'shadow': True,
        })

    bq.insert_rows_json(table_id, out)
    print(f"Wrote {len(out)} audience expansion candidates to {table_id}")


if __name__ == '__main__':
    main()

