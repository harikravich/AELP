#!/usr/bin/env python3
"""
Seed synthetic bidding events into BigQuery for the Auctions Monitor.
Tables: bidding_events_per_minute, bidding_events
"""
import os, random
from datetime import datetime, timedelta, timezone
from google.cloud import bigquery  # type: ignore

PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT')
DATASET = os.getenv('BIGQUERY_TRAINING_DATASET')
assert PROJECT and DATASET

client = bigquery.Client(project=PROJECT)
ds = f"{PROJECT}.{DATASET}"

# Ensure schemas (events table; per-minute view is created via create_bq_views)
client.query(f"""
CREATE TABLE IF NOT EXISTS `{ds}.bidding_events` (
  timestamp TIMESTAMP,
  bid_amount FLOAT64,
  price_paid FLOAT64,
  won BOOL,
  episode_id STRING,
  step INT64
) PARTITION BY DATE(timestamp)
""")

now = datetime.now(timezone.utc)
event_rows = []
for i in range(24*60):
  ts = now - timedelta(minutes=24*60 - i)
  auctions = random.randint(50, 250)
  win_rate = random.uniform(0.25, 0.55)
  wins = int(auctions * win_rate)
  avg_bid = random.uniform(0.5, 2.0)
  avg_price = avg_bid * random.uniform(0.7, 0.95)
  # Few sample events
  for j in range(5):
    bid = max(0.1, random.gauss(avg_bid, 0.2))
    won = random.random() < win_rate
    price = bid * random.uniform(0.6, 0.95) if won else 0.0
    event_rows.append({
      'timestamp': ts.isoformat(),
      'bid_amount': float(bid),
      'price_paid': float(price),
      'won': won,
      'episode_id': f'ep-{ts.date()}',
      'step': random.randint(1, 1000),
    })

client.insert_rows_json(f"{ds}.bidding_events", event_rows)
print(f"Seeded {len(event_rows)} events; refresing per-minute view via create_bq_views")
import subprocess
subprocess.run(["python3", "-m", "AELP2.pipelines.create_bq_views"], check=False)
