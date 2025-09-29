Episode trends
- Win rate trend: SELECT DATE(timestamp) d, AVG(win_rate) FROM `${BIGQUERY_TRAINING_DATASET}.training_episodes` GROUP BY d ORDER BY d;
- CAC trend: SELECT DATE(timestamp) d, SAFE_DIVIDE(SUM(spend), NULLIF(SUM(conversions),0)) cac FROM `${BIGQUERY_TRAINING_DATASET}.training_episodes` GROUP BY d ORDER BY d;
- ROAS trend: SELECT DATE(timestamp) d, SAFE_DIVIDE(SUM(revenue), NULLIF(SUM(spend),0)) roas FROM `${BIGQUERY_TRAINING_DATASET}.training_episodes` GROUP BY d ORDER BY d;

Safety events
- By type: SELECT DATE(timestamp) d, event_type, COUNT(*) FROM `${BIGQUERY_TRAINING_DATASET}.safety_events` GROUP BY d, event_type ORDER BY d;
- Critical recent: SELECT * FROM `${BIGQUERY_TRAINING_DATASET}.safety_events` WHERE severity='critical' AND timestamp > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY);

Model telemetry
- Epsilon distribution: SELECT APPROX_QUANTILES(epsilon, 101) FROM `${BIGQUERY_TRAINING_DATASET}.training_episodes`;
- Avg CPC by model_version: SELECT model_version, AVG(avg_cpc) FROM `${BIGQUERY_TRAINING_DATASET}.training_episodes` GROUP BY model_version ORDER BY 2 DESC;

A/B experiments
- Recent experiments: SELECT * FROM `${BIGQUERY_TRAINING_DATASET}.ab_experiments` ORDER BY timestamp DESC LIMIT 100;
- Variant counts: SELECT variant, COUNT(*) FROM `${BIGQUERY_TRAINING_DATASET}.ab_experiments` GROUP BY variant ORDER BY 2 DESC;

Subagents
- Daily proposals by subagent: SELECT * FROM `${BIGQUERY_TRAINING_DATASET}.subagents_daily` ORDER BY date DESC;
