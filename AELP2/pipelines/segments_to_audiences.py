#!/usr/bin/env python3
"""
Segments → Audiences mapping (shadow-only).

Reads latest `segment_scores_daily`, selects top-N segments, and maps to
platform audience keys (generic placeholders). Writes rows to
`<project>.<dataset>.segment_audience_map` with shadow=true and rationale.

If dependencies/tables are missing, creates the table and exits gracefully.
"""
from __future__ import annotations

import os
from typing import List, Dict
from datetime import datetime

from google.cloud import bigquery
from google.cloud.exceptions import NotFound


def ensure_table(bq: bigquery.Client, project: str, dataset: str) -> str:
    table_id = f"{project}.{dataset}.segment_audience_map"
    try:
        bq.get_table(table_id)
        return table_id
    except NotFound:
        schema = [
            bigquery.SchemaField('date', 'DATE', mode='REQUIRED'),
            bigquery.SchemaField('segment', 'STRING', mode='REQUIRED'),
            bigquery.SchemaField('platform', 'STRING', mode='REQUIRED'),
            bigquery.SchemaField('audience_key', 'STRING', mode='REQUIRED'),
            bigquery.SchemaField('audience_name', 'STRING', mode='NULLABLE'),
            bigquery.SchemaField('rationale', 'STRING', mode='NULLABLE'),
            bigquery.SchemaField('shadow', 'BOOL', mode='REQUIRED'),
            bigquery.SchemaField('created_at', 'TIMESTAMP', mode='REQUIRED'),
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

    # Ensure segment_scores_daily exists
    scores_tbl = f"{project}.{dataset}.segment_scores_daily"
    try:
        bq.get_table(scores_tbl)
    except NotFound:
        print('segment_scores_daily not found; created mapping table only')
        return

    # Read latest date rows and pick top-N segments
    sql = f"""
      WITH last_day AS (
        SELECT MAX(date) AS d FROM `{scores_tbl}`
      )
      SELECT s.date, s.segment, s.score
      FROM `{scores_tbl}` s, last_day
      WHERE s.date = last_day.d
      ORDER BY s.score DESC
      LIMIT 10
    """
    rows = [dict(r) for r in bq.query(sql).result()]
    if not rows:
        print('No segment scores found for latest date; nothing to map')
        return
    now = datetime.utcnow().isoformat()
    out: List[Dict] = []
    for r in rows:
        seg = str(r['segment'])
        key = 'aud_' + ''.join(ch for ch in seg.lower() if ch.isalnum() or ch == '_')[:48]
        out.append({
            'date': r['date'].isoformat(),
            'segment': seg,
            'platform': 'google_ads',
            'audience_key': key,
            'audience_name': seg.title(),
            'rationale': f"Top segment by uplift score ({float(r['score']):.4f}); mapping shadow-only",
            'shadow': True,
            'created_at': now,
        })
    if out:
        bq.insert_rows_json(table_id, out)
        print(f"Wrote {len(out)} segment-audience mappings to {table_id}")
    else:
        print('No mappings produced')


if __name__ == '__main__':
    main()

