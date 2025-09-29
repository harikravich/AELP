#!/usr/bin/env python3
"""
Create calibration stratified views (by channel/device) for RL vs Ads.

Views created when source tables exist:
- `${dataset}.calibration_rl_by_channel_device`
- `${dataset}.calibration_ads_by_channel_device`

Idempotent; supports --dry_run. No data writes.
"""
import os
import argparse
import logging

from AELP2.core.ingestion.bq_loader import get_bq_client

try:
    from google.cloud import bigquery
except Exception as e:  # pragma: no cover
    raise ImportError(f"google-cloud-bigquery required: {e}")

logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO'))
logger = logging.getLogger(__name__)


def create_or_replace_view(client: bigquery.Client, view_id: str, sql: str):
    v = bigquery.Table(view_id)
    v.view_query = sql
    try:
        client.delete_table(view_id, not_found_ok=True)
    except Exception:
        pass
    client.create_table(v)
    logger.info(f"Created view {view_id}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry_run', action='store_true')
    args = ap.parse_args()
    project = os.getenv('GOOGLE_CLOUD_PROJECT')
    dataset = os.getenv('BIGQUERY_TRAINING_DATASET')
    if not project or not dataset:
        if args.dry_run:
            print('[dry_run] Missing project/dataset; would create calibration stratified views')
            return
        raise RuntimeError('Set GOOGLE_CLOUD_PROJECT and BIGQUERY_TRAINING_DATASET')

    if args.dry_run:
        print('[dry_run] would create: calibration_rl_by_channel_device, calibration_ads_by_channel_device')
        return

    client = get_bq_client()
    ds = f"{project}.{dataset}"

    # RL stratified by channel/device (assuming training_episodes has optional fields)
    try:
        client.get_table(f"{ds}.training_episodes")
        rl_sql = f"""
            SELECT
              DATE(timestamp) AS date,
              COALESCE(step_details.channel, 'unknown') AS channel,
              COALESCE(step_details.device, 'unknown') AS device,
              AVG(win_rate) AS avg_win_rate,
              COUNT(*) AS rows
            FROM `{ds}.training_episodes`
            GROUP BY date, channel, device
            ORDER BY date
        """
        create_or_replace_view(client, f"{ds}.calibration_rl_by_channel_device", rl_sql)
    except Exception as e:
        logger.info(f"training_episodes unavailable or lacks fields; skipping RL stratified view: {e}")

    # Ads stratified by channel/device (from ads_campaign_performance if fields exist)
    try:
        client.get_table(f"{ds}.ads_campaign_performance")
        ads_sql = f"""
            SELECT
              DATE(date) AS date,
              COALESCE(channel_type, 'unknown') AS channel,
              COALESCE(device, 'unknown') AS device,
              APPROX_QUANTILES(impression_share, 100)[OFFSET(50)] AS impression_share_p50,
              COUNT(*) AS rows
            FROM `{ds}.ads_campaign_performance`
            GROUP BY date, channel, device
            ORDER BY date
        """
        create_or_replace_view(client, f"{ds}.calibration_ads_by_channel_device", ads_sql)
    except Exception as e:
        logger.info(f"ads_campaign_performance unavailable or lacks fields; skipping Ads stratified view: {e}")


if __name__ == '__main__':
    main()

