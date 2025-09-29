#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "$ROOT_DIR"

if [[ -f .env ]]; then set -a; source ./.env; set +a; fi

echo "[overnight] Loading promo calendar"
python3 AELP2/pipelines/load_promo_calendar.py || true

echo "[overnight] Building triggered (delayed-reward) series"
python3 AELP2/pipelines/build_triggered_series.py || true
echo "[overnight] Building affiliate-triggered series"
python3 AELP2/pipelines/build_affiliate_triggered.py || true
echo "[overnight] Building Impact partner domain map"
python3 AELP2/pipelines/build_impact_partner_domains.py || true

echo "[overnight] Refreshing views"
python3 AELP2/pipelines/create_bq_views.py || true

# Impact.com ingest (if creds provided)
if [[ -n "${IMPACT_ACCOUNT_SID:-}" && ( -n "${IMPACT_AUTH_TOKEN:-}" || -n "${IMPACT_BEARER_TOKEN:-}" ) ]]; then
  echo "[overnight] Ingesting Impact.com partner performance (last 35 days)"
  REPORT_ID=${IMPACT_REPORT_ID:-adv_performance_by_ad_media_date}
  START=$(date -d '35 days ago' +%F)
  END=$(date +%F)
  # Ensure useful SHOW_* flags; caller can override with IMPACT_EXTRA_PARAMS
  export IMPACT_EXTRA_PARAMS=${IMPACT_EXTRA_PARAMS:-"SHOW_DATE=1,SHOW_MEDIA=1,SHOW_TOTAL_CLICKS=1,SHOW_ACTION_TRACKER=1"}
  python3 AELP2/pipelines/impact_run_report_to_bq.py --report "$REPORT_ID" --start "$START" --end "$END" --event Sale || true
  echo "[overnight] Ingesting Impact entities (partners, ads, deals, invoices)"
  python3 AELP2/pipelines/impact_entities_to_bq.py --entities media_partners,ads,deals,invoices,tvr || true
  echo "[overnight] Backfilling Impact daily performance (12 months, auto-discovery)"
  python3 -m AELP2.pipelines.impact_backfill_performance --months 12 || true
else
  echo "[overnight] Impact.com creds not set; skipping impact ingest"
fi

# External affiliate ACH costs (optional CSV)
echo "[overnight] Loading external affiliate ACH costs (if CSV present)"
python3 AELP2/pipelines/load_affiliate_ach_costs.py || true

echo "[overnight] Generating QS alerts & fix tickets"
python3 AELP2/pipelines/generate_qs_alerts.py || true
python3 AELP2/pipelines/generate_qs_fix_tickets.py || true

echo "[overnight] Fitting MMM on triggered KPI"
export AELP2_MMM_SOURCE_VIEW=triggered_kpi_daily
export AELP2_MMM_CHANNEL_LABEL=all_paid_triggered
python3 AELP2/pipelines/mmm_service.py --start "$(date -d '90 days ago' +%F)" --end "$(date +%F)" || true

echo "[overnight] Fitting MMM on Impact only (placeholder: GA enrollments vs Impact payout)"
export AELP2_MMM_SOURCE_VIEW=impact_kpi_daily
export AELP2_MMM_CHANNEL_LABEL=impact
python3 AELP2/pipelines/mmm_service.py --start "$(date -d '120 days ago' +%F)" --end "$(date +%F)" || true

echo "[overnight] Fitting MMM on blended paid (Google + Impact)"
export AELP2_MMM_SOURCE_VIEW=blended_paid_kpi_daily
export AELP2_MMM_CHANNEL_LABEL=paid_blended
python3 AELP2/pipelines/mmm_service.py --start "$(date -d '120 days ago' +%F)" --end "$(date +%F)" || true

echo "[overnight] Fitting MMM on Search brand/nonbrand (directional)"
export AELP2_MMM_SOURCE_VIEW=search_brand_daily
export AELP2_MMM_CHANNEL_LABEL=google_brand
python3 AELP2/pipelines/mmm_service.py --start "$(date -d '60 days ago' +%F)" --end "$(date +%F)" || true
export AELP2_MMM_SOURCE_VIEW=search_nonbrand_daily
export AELP2_MMM_CHANNEL_LABEL=google_nonbrand
python3 AELP2/pipelines/mmm_service.py --start "$(date -d '60 days ago' +%F)" --end "$(date +%F)" || true

echo "[overnight] GSC brand share (best-effort)"
python3 AELP2/pipelines/gsc_to_bq.py || true

echo "[overnight] Done."
