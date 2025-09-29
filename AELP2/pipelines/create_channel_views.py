#!/usr/bin/env python3
"""
Create channel-specific daily views from ads_campaign_performance using advertising_channel_type.

Views created (if column exists):
- google_search_campaign_daily
- google_video_campaign_daily
- google_discovery_campaign_daily
- google_pmax_campaign_daily
"""
import os
from google.cloud import bigquery


def create_view(bq: bigquery.Client, view_id: str, sql: str):
    try:
        bq.delete_table(view_id, not_found_ok=True)
    except Exception:
        pass
    view = bigquery.Table(view_id)
    view.view_query = sql
    bq.create_table(view)


def main():
    project = os.getenv('GOOGLE_CLOUD_PROJECT')
    dataset = os.getenv('BIGQUERY_TRAINING_DATASET')
    if not project or not dataset:
        raise RuntimeError('Set GOOGLE_CLOUD_PROJECT and BIGQUERY_TRAINING_DATASET')
    bq = bigquery.Client(project=project)
    base = f"{project}.{dataset}"
    # Check column presence
    table = bq.get_table(f"{base}.ads_campaign_performance")
    fields = {f.name for f in table.schema}
    if 'advertising_channel_type' not in fields:
        print('advertising_channel_type not in ads_campaign_performance; skipping channel views')
        return
    tmpl = lambda ch: f"""
      SELECT DATE(date) AS date,
             SUM(impressions) AS impressions,
             SUM(clicks) AS clicks,
             SUM(cost_micros)/1e6 AS cost,
             SUM(conversions) AS conversions,
             SUM(conversion_value) AS revenue,
             SAFE_DIVIDE(SUM(cost_micros)/1e6, NULLIF(SUM(conversions),0)) AS cac,
             SAFE_DIVIDE(SUM(conversion_value), NULLIF(SUM(cost_micros)/1e6,0)) AS roas
      FROM `{base}.ads_campaign_performance`
      WHERE advertising_channel_type = '{ch}'
      GROUP BY date
      ORDER BY date
    """
    create_view(bq, f"{base}.google_search_campaign_daily", tmpl('SEARCH'))
    create_view(bq, f"{base}.google_video_campaign_daily", tmpl('VIDEO'))
    create_view(bq, f"{base}.google_discovery_campaign_daily", tmpl('DISCOVERY'))
    create_view(bq, f"{base}.google_pmax_campaign_daily", tmpl('PERFORMANCE_MAX'))
    print('Created channel views for Google Ads')


if __name__ == '__main__':
    main()

