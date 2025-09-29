# Data Contracts (Versioned)

Scope: `${BIGQUERY_TRAINING_DATASET}` tables and core views.

Contract v1 (2025-09-07)
- training_episodes(timestamp DATE partition), required fields: spend, revenue, conversions, win_rate
- safety_events (v2): timestamp, event_type, severity, metadata JSON
- canary_changes: timestamp partitioned, run_id, campaign_id, old_budget, new_budget, delta_pct, shadow, applied
- ab_experiments: start (partition), experiment_id, platform, campaign_id, status
- creative_variants: created (partition), variant_id, experiment_id, gen_method, text, policy_flags
- ops_flow_runs: timestamp (partition), flow, rc_map JSON, failures JSON, ok

Evolution:
- Additive only within v1; new fields nullable.
- Breaking changes require v2 with migration guidance.

