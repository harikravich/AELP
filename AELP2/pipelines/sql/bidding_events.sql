-- Bidding Events schema (R&D dataset)
-- Env: set `GOOGLE_CLOUD_PROJECT` and `BIGQUERY_TRAINING_DATASET`, then run via bq CLI:
-- bq query --use_legacy_sql=false < AELP2/pipelines/sql/bidding_events.sql

DECLARE ds STRING DEFAULT @dataset;
DECLARE project STRING DEFAULT @project;

-- Replace variables if bq parameters not used
SET ds = IFNULL(ds, '${BIGQUERY_TRAINING_DATASET}');
SET project = IFNULL(project, '${GOOGLE_CLOUD_PROJECT}');

EXECUTE IMMEDIATE FORMAT("""
CREATE SCHEMA IF NOT EXISTS `%s.%s`
""", project, ds);

EXECUTE IMMEDIATE FORMAT("""
CREATE TABLE IF NOT EXISTS `%s.%s.bidding_events` (
  timestamp TIMESTAMP NOT NULL,
  episode_id STRING,
  step INT64,
  user_id STRING,
  campaign_id STRING,
  bid_amount FLOAT64 NOT NULL,
  won BOOL,
  price_paid FLOAT64,
  auction_id STRING,
  context JSON,
  explain JSON
) PARTITION BY DATE(timestamp)
""", project, ds);

EXECUTE IMMEDIATE FORMAT("""
CREATE OR REPLACE VIEW `%s.%s.bidding_events_per_minute` AS
SELECT
  TIMESTAMP_TRUNC(timestamp, MINUTE) AS minute,
  COUNT(*) AS auctions,
  COUNTIF(won) AS wins,
  SAFE_DIVIDE(COUNTIF(won), NULLIF(COUNT(*),0)) AS win_rate,
  AVG(bid_amount) AS avg_bid,
  AVG(price_paid) AS avg_price_paid
FROM `%s.%s.bidding_events`
WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
GROUP BY minute
ORDER BY minute
""", project, ds, project, ds);

