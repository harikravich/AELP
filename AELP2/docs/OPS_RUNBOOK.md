Ops Runbook

Daily
- Verify nightly jobs completed (logs for AELP2/scripts/nightly_jobs.sh)
- Check BigQuery freshness (training_episodes_daily, ads_campaign_daily, ga4_daily)
- Review safety_events for spikes (gate violations)
- Review fidelity_evaluations (status, metrics)

New Jobs (P0→P2)
- Creative Publisher (manual trigger or scheduler disabled by default)
  - Script: `python -m AELP2.pipelines.publish_google_creatives` (paused publish)
  - Tables: `creative_publish_queue`, `creative_publish_log`
  - Flags: `GATES_ENABLED=1`, `AELP2_ALLOW_BANDIT_MUTATIONS=1`, `ALLOW_REAL_CREATIVE=1` for live (keep 0 initially)
- LP Module Runner (background)
  - Script: `python -m AELP2.pipelines.module_runner`
  - Reads queued runs; calls connectors; writes `module_results`; short retention
  - Flags: `AELP2_LP_MODULES_ENABLED`, per‑module LIVE flags
- A/B Aggregation
  - Populates `ab_metrics_daily` from Ads + GA4 joins; runs nightly
- Bandit Posteriors Writer
  - Writes `${DATASET}.bandit_posteriors` and snapshots policies; runs daily
- GeoLift / Halo Reads (weekly)
  - R container job; writes `halo_reads_daily`; update Ramp panel
- dbt Models + GE DQ
  - dbt build for canonical views; Great Expectations suite; gate alerts to `ops_alerts`

On-call Alerts (to configure in Cloud Monitoring)
- Ingestion failures (MCC jobs, GA4)
- Telemetry writes failures (training_episodes)
- Safety spike: gate violations > threshold in last hour
- Spend velocity anomaly vs pacer

Emergency Stop
- Trigger via safety.emergency_stop (API/UI) with reason; verify campaigns are paused in adapters
- CLI: `python -m AELP2.scripts.emergency_stop --reason "manual_stop" --note "drill"`
- Notify stakeholders; capture event in safety_events

Playbooks
- BigQuery insert failures: run schema reconcile (writer will add missing fields), re-run episode write
- Zero wins: increase AELP2_CALIBRATION_FLOOR_RATIO or enable auto-tune; verify calibration
- Approval spam: set AELP2_HITL_MIN_STEP_FOR_APPROVAL=200 and AELP2_HITL_ON_GATE_FAIL_FOR_BIDS=0; enable throttle

Troubleshooting – Creative Publisher
- “Missing Ads env” → set GOOGLE_ADS_* and GOOGLE_ADS_LOGIN_CUSTOMER_ID
- “No YouTube ID” → paste an existing YouTube video ID or disable video type
- “Policy topics” → review reasons; fix copy/assets; re‑enqueue

Troubleshooting – Experiments
- SRM red badge → check assignment mix (namespace/salt, traffic leaks)
- No metric rows → ensure `ab_metrics_daily` job joins Ads/GA4 and KPI IDs are locked

Config References
- Ads creds: .google_ads_credentials.env
- Project/dataset: .env (GOOGLE_CLOUD_PROJECT, BIGQUERY_TRAINING_DATASET)
- HITL: AELP2_HITL_* flags; throttle: AELP2_HITL_THROTTLE_ON_FAIL
- Floor: AELP2_CALIBRATION_FLOOR_RATIO; auto-tune: AELP2_FLOOR_AUTOTUNE_*

Nightly/Scheduled Jobs
- Views refresh: `python -m AELP2.pipelines.create_bq_views`
- MCC ingestion: `bash AELP2/scripts/run_ads_ingestion.sh --mcc <ID> --last14 --tasks "campaigns,ad_performance,keywords,search_terms,geo_device,adgroups,conversion_actions"`
- Fidelity eval: `bash AELP2/scripts/run_fidelity.sh`
- DQ checks: `bash AELP2/scripts/run_dq_checks.sh`

Backfills (large windows)
- 3‑year Ads backfill (monthly chunks, MCC):
  - `bash AELP2/scripts/run_ads_backfill.sh --mcc <ID> --tasks "campaigns,ad_performance,keywords,search_terms,geo_device,adgroups,conversion_actions"`
  - Dry‑run first to review plan: add `--dry-run`
  - Tune: `AELP2_ADS_MCC_DELAY_SECONDS` (between accounts), `AELP2_BACKFILL_SLEEP_BETWEEN_WINDOWS` (between months)
  - Monitor quotas/errors in logs; rerun failed windows as needed
