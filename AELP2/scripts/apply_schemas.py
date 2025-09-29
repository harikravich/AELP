#!/usr/bin/env python3
"""Apply core BigQuery schemas (idempotent)."""
import os
from google.cloud import bigquery  # type: ignore

PROJECT = os.environ.get('GOOGLE_CLOUD_PROJECT')
DATASET = os.environ.get('BIGQUERY_TRAINING_DATASET')
USERS = os.environ.get('BIGQUERY_USERS_DATASET', 'gaelp_users')

DDL = [
    # Experiments
    f"CREATE TABLE IF NOT EXISTS `{PROJECT}.{DATASET}.ab_assignments` (timestamp TIMESTAMP, experiment STRING, variant STRING, unit_id STRING, unit_type STRING, context JSON) PARTITION BY DATE(timestamp) CLUSTER BY experiment, variant, unit_id",
    f"CREATE TABLE IF NOT EXISTS `{PROJECT}.{DATASET}.ab_metrics_daily` (date DATE, experiment STRING, variant STRING, spend FLOAT64, clicks INT64, conversions INT64, revenue FLOAT64, cost FLOAT64, cac FLOAT64, roas FLOAT64) PARTITION BY date CLUSTER BY experiment, variant",
    # Explore & RL
    f"CREATE TABLE IF NOT EXISTS `{PROJECT}.{DATASET}.explore_cells` (cell_key STRING, angle STRING, audience STRING, channel STRING, lp STRING, offer STRING, last_seen TIMESTAMP, spend FLOAT64, clicks INT64, conversions INT64, revenue FLOAT64, cac FLOAT64, value FLOAT64) PARTITION BY DATE(last_seen)",
    f"CREATE TABLE IF NOT EXISTS `{PROJECT}.{DATASET}.bandit_posteriors` (ts TIMESTAMP, cell_key STRING, metric STRING, mean FLOAT64, ci_low FLOAT64, ci_high FLOAT64, samples INT64) PARTITION BY DATE(ts)",
    # Creative
    f"CREATE TABLE IF NOT EXISTS `{PROJECT}.{DATASET}.creative_publish_queue` (enqueued_at TIMESTAMP, run_id STRING, platform STRING, type STRING, campaign_id STRING, ad_group_id STRING, asset_group_id STRING, payload JSON, status STRING, requested_by STRING) PARTITION BY DATE(enqueued_at)",
    f"CREATE TABLE IF NOT EXISTS `{PROJECT}.{DATASET}.creative_publish_log` (ts TIMESTAMP, run_id STRING, platform STRING, platform_ids JSON, status STRING, policy_topics JSON, error STRING) PARTITION BY DATE(ts)",
    # LP
    f"CREATE TABLE IF NOT EXISTS `{PROJECT}.{DATASET}.lp_tests` (test_id STRING, created_at TIMESTAMP, lp_a STRING, lp_b STRING, status STRING, traffic_split FLOAT64, primary_metric STRING) PARTITION BY DATE(created_at)",
    f"CREATE TABLE IF NOT EXISTS `{PROJECT}.{DATASET}.funnel_dropoffs` (date DATE, lp_url STRING, stage STRING, visitors INT64, drop_rate FLOAT64) PARTITION BY date",
    f"CREATE TABLE IF NOT EXISTS `{PROJECT}.{DATASET}.lp_module_runs` (run_id STRING, slug STRING, page_url STRING, consent_id STRING, created_ts TIMESTAMP, status STRING, elapsed_ms INT64, error_code STRING) PARTITION BY DATE(created_ts)",
    f"CREATE TABLE IF NOT EXISTS `{PROJECT}.{DATASET}.consent_logs` (consent_id STRING, slug STRING, page_url STRING, consent_text STRING, ip_hash STRING, user_agent STRING, ts TIMESTAMP) PARTITION BY DATE(ts)",
    f"CREATE TABLE IF NOT EXISTS `{PROJECT}.{DATASET}.module_results` (run_id STRING, slug STRING, summary_text STRING, result_json JSON, expires_at TIMESTAMP) PARTITION BY DATE(expires_at)",
    # Halo
    f"CREATE TABLE IF NOT EXISTS `{PROJECT}.{DATASET}.halo_experiments` (exp_id STRING, channel STRING, geo STRING, start DATE, end_date DATE, treatment_share FLOAT64, status STRING) PARTITION BY start",
    f"CREATE TABLE IF NOT EXISTS `{PROJECT}.{DATASET}.halo_reads_daily` (date DATE, exp_id STRING, brand_lift FLOAT64, ci_low FLOAT64, ci_high FLOAT64, method STRING) PARTITION BY date",
    f"CREATE TABLE IF NOT EXISTS `{PROJECT}.{DATASET}.channel_interference_scores` (date DATE, from_channel STRING, to_channel STRING, cannibalization FLOAT64, lift FLOAT64) PARTITION BY date",
    # Research
    f"CREATE TABLE IF NOT EXISTS `{PROJECT}.{DATASET}.channel_candidates` (id STRING, name STRING, type STRING, status STRING, audience_fit_notes STRING, use_cases ARRAY<STRING>, pricing_model STRING, typical_cpc FLOAT64, min_budget FLOAT64, formats ARRAY<STRING>, targeting JSON, api_available BOOL, docs_url STRING, auth_type STRING, export_mode STRING, measurability JSON, risk_notes STRING, bot_fraud_risk STRING, effort_estimate STRING, integration_steps JSON, score_fit FLOAT64, score_cost FLOAT64, score_measure FLOAT64, score_effort FLOAT64, score_risk FLOAT64, score_total FLOAT64, citations ARRAY<JSON>, created_by STRING, created_at TIMESTAMP, updated_at TIMESTAMP) PARTITION BY DATE(created_at)",
    f"CREATE TABLE IF NOT EXISTS `{PROJECT}.{DATASET}.research_findings` (ts TIMESTAMP, candidate_id STRING, summary STRING, details JSON, citations ARRAY<JSON>, source STRING) PARTITION BY DATE(ts)",
    # Users dataset
    f"CREATE TABLE IF NOT EXISTS `{PROJECT}.{USERS}.canvas_pins` (ts TIMESTAMP, user_email STRING, pin_id STRING, payload JSON) PARTITION BY DATE(ts)",
]

def main():
    assert PROJECT and DATASET
    bq = bigquery.Client(project=PROJECT)
    for ddl in DDL:
        bq.query(ddl).result()
    print('Schemas applied.')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
