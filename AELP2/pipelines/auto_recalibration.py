#!/usr/bin/env python3
"""
Auto-recalibration (shadow-only): detect drift and log proposals.

Logic:
- Read last 14 days from `${dataset}.fidelity_evaluations`.
- If KS or error metrics exceed thresholds (env or defaults), write a safety event
  and insert a proposal into `${dataset}.calibration_proposals` with recommended action.

Never mutates live parameters. HITL required to apply any change elsewhere.
"""
import os
import json
from datetime import datetime, timedelta, timezone

try:
    from google.cloud import bigquery
    from google.cloud.exceptions import NotFound
except Exception as e:  # pragma: no cover
    raise ImportError(f"google-cloud-bigquery required: {e}")


def _ensure_tables(bq: bigquery.Client, project: str, dataset: str):
    proposals = f"{project}.{dataset}.calibration_proposals"
    safety = f"{project}.{dataset}.safety_events"
    # proposals
    try:
        bq.get_table(proposals)
    except NotFound:
        schema = [
            bigquery.SchemaField('timestamp', 'TIMESTAMP'),
            bigquery.SchemaField('window_days', 'INT64'),
            bigquery.SchemaField('reason', 'STRING'),
            bigquery.SchemaField('recommendation', 'STRING'),
            bigquery.SchemaField('metadata', 'JSON'),
            bigquery.SchemaField('shadow', 'BOOL'),
        ]
        t = bigquery.Table(proposals, schema=schema)
        t.time_partitioning = bigquery.TimePartitioning(type_=bigquery.TimePartitioningType.DAY, field='timestamp')
        bq.create_table(t)
    # safety_events (v2) if absent create minimal v2-compatible table
    try:
        bq.get_table(safety)
    except NotFound:
        schema = [
            bigquery.SchemaField('timestamp', 'TIMESTAMP'),
            bigquery.SchemaField('event_type', 'STRING'),
            bigquery.SchemaField('severity', 'STRING'),
            bigquery.SchemaField('episode_id', 'STRING'),
            bigquery.SchemaField('metadata', 'JSON'),
            bigquery.SchemaField('action_taken', 'STRING'),
        ]
        t = bigquery.Table(safety, schema=schema)
        t.time_partitioning = bigquery.TimePartitioning(type_=bigquery.TimePartitioningType.DAY, field='timestamp')
        bq.create_table(t)
    return proposals, safety


def main():
    project = os.getenv('GOOGLE_CLOUD_PROJECT')
    dataset = os.getenv('BIGQUERY_TRAINING_DATASET')
    if not project or not dataset:
        print('Missing env; skipping')
        return
    window_days = int(os.getenv('AELP2_RECAL_WINDOW_DAYS', '14'))
    thr_ks = float(os.getenv('AELP2_FIDELITY_MAX_KS_WINRATE', '0.35') or 0.35)
    thr_mse = float(os.getenv('AELP2_FIDELITY_MAX_MSE_ROAS', '1.5') or 1.5)
    bq = bigquery.Client(project=project)
    proposals_tbl, safety_tbl = _ensure_tables(bq, project, dataset)

    # Fetch recent fidelity evaluations
    sql = f"""
      SELECT * FROM `{project}.{dataset}.fidelity_evaluations`
      WHERE DATE(timestamp) BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL {window_days} DAY) AND CURRENT_DATE()
      ORDER BY timestamp DESC
      LIMIT 100
    """
    rows = [dict(r) for r in bq.query(sql).result()]
    if not rows:
        print('No fidelity evaluations found; nothing to do')
        return
    # Simple decision: if any violation over window, propose recalibration
    violate = False
    for r in rows:
        ks = float(r.get('ks_winrate_vs_impressionshare') or 0.0)
        details = r.get('details')
        try:
            det = json.loads(details) if isinstance(details, str) else (details or {})
        except Exception:
            det = {}
        mse_roas = float(det.get('rmse_roas') or r.get('rmse_roas') or 0.0)
        if (ks and ks > thr_ks) or (mse_roas and mse_roas > thr_mse):
            violate = True
            break
    if not violate:
        print('No threshold violations; no proposals')
        return
    now = datetime.now(timezone.utc).isoformat()
    recommendation = 'rebuild_reference_and_adjust_floor_ratio'
    meta = {
        'window_days': window_days,
        'thresholds': {'ks': thr_ks, 'rmse_roas': thr_mse},
        'notes': 'shadow-only; requires HITL to apply',
    }
    # Insert proposal
    bq.insert_rows_json(proposals_tbl, [{
        'timestamp': now,
        'window_days': window_days,
        'reason': 'fidelity_threshold_violation',
        'recommendation': recommendation,
        'metadata': json.dumps(meta),
        'shadow': True,
    }])
    # Write safety event
    bq.insert_rows_json(safety_tbl, [{
        'timestamp': now,
        'event_type': 'calibration_drift',
        'severity': 'WARNING',
        'episode_id': None,
        'metadata': json.dumps({'recommendation': recommendation, 'meta': meta}),
        'action_taken': 'proposal_logged',
    }])
    print('Auto-recalibration proposal logged (shadow) and safety event written')


if __name__ == '__main__':
    main()

