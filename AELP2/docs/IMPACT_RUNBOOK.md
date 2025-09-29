Impact.com → BigQuery → MMM Runbook (Sep 17, 2025)

What’s set up
- Entity ingester (MediaPartners, Ads, Deals, Invoices, TVR): AELP2/pipelines/impact_entities_to_bq.py
- Report exporter (CSV → BQ, daily performance): AELP2/pipelines/impact_to_bq.py
- Nightly pack runs both and MMM fits: AELP2/scripts/run_overnight_pack.sh
- BQ views:
  - impact_partner_daily — from impact_partner_performance if present; else from invoices
  - impact_platform_daily — daily platform cost (payout)
  - cross_platform_kpi_daily — Ads + Impact + GA conversions
  - blended_paid_kpi_daily — total paid cost (Google + Impact) vs GA enrollments
  - ga_affiliate_triggered_daily / _channel_daily — GA touch‑aligned affiliate signal
  - impact_partner_domains — partner→(website domains, roots) map
  - ga_affiliate_triggered_by_partner_daily — affiliate triggered mapped to partners (heuristic by domain root)
  - affiliate_brand_lift_by_partner — best lag corr(triggered_partner vs brand search conversions)

How to run ad hoc
```bash
# Full entities backfill
python3 AELP2/pipelines/impact_entities_to_bq.py --entities media_partners,ads,deals,invoices,tvr

# Try a report export (adjust report ID)
python3 AELP2/pipelines/impact_to_bq.py --list | head -n 50
python3 AELP2/pipelines/impact_to_bq.py --report adv_performance_by_ad_media_date --start 2025-01-01 --end 2025-09-16

# Rebuild views
python3 -m AELP2.pipelines.create_bq_views

# MMM blended (paid)
export AELP2_MMM_SOURCE_VIEW=blended_paid_kpi_daily
export AELP2_MMM_CHANNEL_LABEL=paid_blended
python3 AELP2/pipelines/mmm_service.py --start 2025-05-01 --end 2025-09-16

# Optional: Bayesian bootstrap curves (writes model=bayesian_bootstrap)
python3 AELP2/pipelines/mmm_service.py --start 2025-05-01 --end 2025-09-16 \
  --model_label bayesian_bootstrap --bootstrap 200
```

What still needs UI toggle in Impact
- Pick a daily performance report (e.g., “Performance by Ad / Partner / Date”) and ensure “API Accessible” is ON.
- If your account uses Program/SubID, set IMPACT_SUBAID in `.env` to scope data.
  As soon as a daily report is accessible, the loader will populate:
  - gaelp_training.impact_partner_performance (daily aggregate)
  - Optional gaelp_training.impact_actions_raw (row level)
  - Views will automatically prefer daily performance over the invoice fallback.

Checks
- Entities row counts: MediaPartners≈1.9k, Ads≈1.0k, Invoices≈3.5k.
- Monthly payout parity: `impact_platform_daily` sum is within ~2–9% of invoice totals (based on CreatedDate). If discrepancy >10%, review which invoice fields to include.

Dashboards
- Affiliates page (8080): /affiliates — top partners by payout/actions, filters for lookback/min payout.
- Status: /api/control/status includes Impact last dates.

Affiliate joiners (current heuristic)
- We map GA affiliate sources (src/med) to Impact partners via partner website domains (impact_partner_domains.root).
- View `ga_affiliate_triggered_by_partner_daily` uses REGEXP_CONTAINS on the GA source with each partner’s domain root.
- Once daily performance/actions land, replace with deterministic joins via click ids/UTMs where available.
