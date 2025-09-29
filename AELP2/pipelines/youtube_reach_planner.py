#!/usr/bin/env python3
"""
YouTube Reach Planner — real API when available.

Writes estimated reach metrics to BigQuery. Attempts to call Google Ads
ReachPlanService when google-ads SDK and credentials are present; otherwise
falls back to a safe stub row with a clear note.
"""
import os
from datetime import date
from google.cloud import bigquery
from google.cloud.exceptions import NotFound


def ensure_table(bq: bigquery.Client, project: str, dataset: str) -> str:
    table_id = f"{project}.{dataset}.youtube_reach_estimates"
    try:
        bq.get_table(table_id)
        return table_id
    except NotFound:
        schema = [
            bigquery.SchemaField('date', 'DATE', mode='REQUIRED'),
            bigquery.SchemaField('inventory', 'STRING', mode='REQUIRED'),
            bigquery.SchemaField('est_impressions', 'INT64', mode='NULLABLE'),
            bigquery.SchemaField('est_unique_reach', 'INT64', mode='NULLABLE'),
            bigquery.SchemaField('cpm', 'FLOAT', mode='NULLABLE'),
            bigquery.SchemaField('notes', 'STRING', mode='NULLABLE'),
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
    # Attempt API import; fallback to stub
    note = 'google-ads not installed; writing stub estimate'
    inv = os.getenv('AELP2_YT_INVENTORY', 'YOUTUBE')
    impressions = 100000
    reach = 60000
    cpm = 9.50
    try:
        from google.ads.googleads.client import GoogleAdsClient  # type: ignore
        cfg = {
            'developer_token': os.environ['GOOGLE_ADS_DEVELOPER_TOKEN'],
            'client_id': os.environ['GOOGLE_ADS_CLIENT_ID'],
            'client_secret': os.environ['GOOGLE_ADS_CLIENT_SECRET'],
            'refresh_token': os.environ['GOOGLE_ADS_REFRESH_TOKEN'],
            'use_proto_plus': True,
        }
        login_cid = os.getenv('GOOGLE_ADS_LOGIN_CUSTOMER_ID')
        if login_cid:
            cfg['login_customer_id'] = login_cid.replace('-', '')
        client = GoogleAdsClient.load_from_dict(cfg)
        service = client.get_service('ReachPlanService')
        # Location: US by default (2840); can set via env AELP2_YT_LOCATION_ID
        loc = int(os.getenv('AELP2_YT_LOCATION_ID', '2840'))
        # Currency from env or default USD
        currency = os.getenv('AELP2_YT_CURRENCY', 'USD')
        # Budget in micros from env (default $10k)
        budget_micros = int(float(os.getenv('AELP2_YT_BUDGET', '10000')) * 1e6)
        # Duration in days
        dur = int(os.getenv('AELP2_YT_DURATION_DAYS', '28'))
        products = [client.get_type('PlannedProductAllocation')]
        products[0].plannable_product_code = inv
        products[0].budget_micros = budget_micros
        req = client.get_type('GenerateReachForecastRequest')
        req.customer_id = (os.getenv('GOOGLE_ADS_CUSTOMER_ID') or '0000000000').replace('-', '')
        req.campaign_duration.duration_in_days = dur
        req.currency_code = currency
        req.min_effective_frequency = 1
        req.targeting.location_id = loc
        req.targeting.plannable_location_id = loc
        req.planned_products.extend(products)
        resp = service.generate_reach_forecast(request=req)
        if resp and resp.reach_curve and resp.reach_curve.reach_forecasts:
            # Take last point on curve
            pt = resp.reach_curve.reach_forecasts[-1]
            impressions = int(pt.on_target_impressions or pt.total_impressions or impressions)
            reach = int(pt.on_target_reach or pt.reach or reach)
            cpm = float(pt.cost_micros / max(pt.total_impressions or 1, 1)) * 1e6 if getattr(pt, 'total_impressions', 0) else cpm
            note = 'reach_planner_ok'
        else:
            note = 'reach_planner_empty_response'
    except Exception as e:
        note = f'reach_planner_error: {e}'
    row = {
        'date': date.today().isoformat(),
        'inventory': 'yt_instream',
        'est_impressions': impressions,
        'est_unique_reach': reach,
        'cpm': cpm,
        'notes': note,
    }
    bq.insert_rows_json(table_id, [row])
    print(f"Reach estimates written to {table_id}")


if __name__ == '__main__':
    main()
