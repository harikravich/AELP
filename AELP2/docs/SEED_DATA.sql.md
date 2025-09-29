# Seed Data Snippets (for Dry Runs)

```sql
-- One A/B experiment and sample assignment
INSERT INTO `${PROJECT}.${DATASET}.ab_experiments` (start, end, experiment_id, platform, campaign_id, status, variants)
VALUES (CURRENT_DATE(), NULL, 'lp_balance_v1', 'web', NULL, 'running', JSON '["A","B"]');

INSERT INTO `${PROJECT}.${DATASET}.ab_assignments`
(timestamp, experiment, variant, unit_id, unit_type, context)
VALUES (CURRENT_TIMESTAMP(), 'lp_balance_v1', 'A', 'client_abc', 'ga4_client_id', JSON '{"source":"seed"}');

-- One explore cell
INSERT INTO `${PROJECT}.${DATASET}.explore_cells` (cell_key, angle, audience, channel, lp, offer, last_seen, spend, clicks, conversions, revenue, cac, value)
VALUES ('balance_safety|parents_ios|search_nb|/balance-v2|trial','balance_safety','parents_ios','search_nb','/balance-v2','trial',CURRENT_TIMESTAMP(),0,0,0,0,NULL,NULL);

-- Creative queue example
INSERT INTO `${PROJECT}.${DATASET}.creative_publish_queue`
(enqueued_at, run_id, platform, type, campaign_id, ad_group_id, asset_group_id, payload, status, requested_by)
VALUES (CURRENT_TIMESTAMP(), 'run_seed_1','google_ads','rsa','1234567890','0987654321',NULL,
  JSON '{"headlines":["See Online Balance","Safer Gaming Alerts"],"descriptions":["Help your teen thrive online."],"final_url":"https://example.com/balance"}',
  'queued','seed');
```

