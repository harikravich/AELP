#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Nightly: refresh exports (if creds OK), enrich, recompute fidelity, uplift, OPE, and readiness.
(
  set -a; source <(sed -n 's/^export //p' .env 2>/dev/null || true); set +a
  # Optional: refresh placement conversions
  /opt/aelp/venvs/aelp-heavy/bin/python AELP2/tools/fetch_campaign_placement_conversions.py || true
  /opt/aelp/venvs/aelp-heavy/bin/python AELP2/tools/build_placement_calibrators.py || true
  /opt/aelp/venvs/aelp-heavy/bin/python AELP2/tools/enrich_creatives_with_objects.py || true
  /opt/aelp/venvs/aelp-heavy/bin/python AELP2/tools/hourly_multipliers.py || true
  # Config snapshots for policy flags
  /opt/aelp/venvs/aelp-heavy/bin/python AELP2/tools/export_meta_ad_configs.py || true
  AELP2_DR=1 AELP2_DR_SCALE=0.08 /opt/aelp/venvs/aelp-heavy/bin/python AELP2/tools/sim_fidelity_campaigns_temporal_v3.py || true
  /opt/aelp/venvs/aelp-heavy/bin/python AELP2/tools/ad_level_accuracy.py || true
  /opt/aelp/venvs/aelp-heavy/bin/python AELP2/tools/uplift_baseline_test.py || true
  /opt/aelp/venvs/aelp-heavy/bin/python AELP2/tools/ope_upgrades.py || true
  # Copy mining (Meta cached + Google Ads + Impact + YouTube) and merge
  /opt/aelp/venvs/aelp-heavy/bin/python AELP2/tools/mine_meta_copy.py || true
  /opt/aelp/venvs/aelp-heavy/bin/python AELP2/tools/pull_google_ads_copy.py || true
  /opt/aelp/venvs/aelp-heavy/bin/python AELP2/tools/pull_google_ads_copy_rest.py || true
  /opt/aelp/venvs/aelp-heavy/bin/python AELP2/tools/pull_impact_copy.py || true
  /opt/aelp/venvs/aelp-heavy/bin/python AELP2/tools/crawl_youtube_titles.py || true
  /opt/aelp/venvs/aelp-heavy/bin/python AELP2/tools/merge_copy_banks.py || true
  /opt/aelp/venvs/aelp-heavy/bin/python AELP2/tools/propose_kb_non_claims.py || true
  # Prefer BQ-backed copy pulls if ADC is configured
  /opt/aelp/venvs/aelp-heavy/bin/python AELP2/tools/pull_google_ads_copy_from_bq.py || true
  /opt/aelp/venvs/aelp-heavy/bin/python AELP2/tools/pull_impact_copy_from_bq.py || true
  # Weekly evaluation (run on Mondays or when forced)
  DOW=$(date +%u)
  if [[ "${AELP2_FORCE_WEEKLY:-0}" = "1" || "$DOW" = "1" ]]; then
    AELP2_BACKFILL_DAYS=${AELP2_BACKFILL_DAYS:-365} /opt/aelp/venvs/aelp-heavy/bin/python AELP2/tools/backfill_ad_daily_insights.py || true
    /opt/aelp/venvs/aelp-heavy/bin/python AELP2/tools/build_weekly_predictions.py || true
    /opt/aelp/venvs/aelp-heavy/bin/python AELP2/tools/annotate_weekly_with_policy.py || true
    /opt/aelp/venvs/aelp-heavy/bin/python AELP2/tools/eval_wbua.py || true
    AELP2_WEEKLY_DIR=AELP2/reports/weekly_creatives_policy /opt/aelp/venvs/aelp-heavy/bin/python AELP2/tools/eval_wbua.py || true
    /opt/aelp/venvs/aelp-heavy/bin/python AELP2/tools/dual_gate_weekly.py || true
    # Enrich weekly with CPC mix and fit calibrator
    /opt/aelp/venvs/aelp-heavy/bin/python AELP2/tools/enrich_weekly_with_cpc.py || true
    /opt/aelp/venvs/aelp-heavy/bin/python AELP2/tools/fit_cac_calibrator.py || true
    AELP2_WEEKLY_DIR=AELP2/reports/weekly_creatives_policy /opt/aelp/venvs/aelp-heavy/bin/python AELP2/tools/dual_gate_weekly.py || true
    /opt/aelp/venvs/aelp-heavy/bin/python AELP2/tools/conformal_topk_weekly.py || true
    /opt/aelp/venvs/aelp-heavy/bin/python AELP2/tools/weekly_topN_dual.py || true
    /opt/aelp/venvs/aelp-heavy/bin/python AELP2/tools/portfolio_selector.py || true
    /opt/aelp/venvs/aelp-heavy/bin/python AELP2/tools/check_dual_gate_thresholds.py || true
    AELP2_WEEKLY_DIR=AELP2/reports/weekly_creatives_policy /opt/aelp/venvs/aelp-heavy/bin/python AELP2/tools/weekly_topN_dual.py || true
    AELP2_WEEKLY_DIR=AELP2/reports/weekly_creatives_policy /opt/aelp/venvs/aelp-heavy/bin/python AELP2/tools/ope_config_weighting.py || true
    /opt/aelp/venvs/aelp-heavy/bin/python AELP2/tools/report_policy_vs_mixed.py || true
  fi
) 2>&1 | tee -a AELP2/reports/nightly.log

echo "Nightly jobs completed. See AELP2/reports for outputs."
