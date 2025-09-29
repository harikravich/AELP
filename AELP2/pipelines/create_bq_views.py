#!/usr/bin/env python3
"""
Create standard BigQuery views for dashboards/subagents.

Views:
- training_episodes_daily
- ads_campaign_daily

Requirements:
- Env: GOOGLE_CLOUD_PROJECT, BIGQUERY_TRAINING_DATASET
"""

import os
import logging
import shutil
import subprocess

try:
    from google.cloud import bigquery
except Exception as e:
    raise ImportError(f"google-cloud-bigquery is required: {e}")

from AELP2.core.ingestion.bq_loader import get_bq_client

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


def _create_or_replace_view_via_cli(project: str, view_id: str, sql: str):
    if shutil.which("bq") is None:
        raise RuntimeError("bq CLI not found for fallback view creation")
    # view_id is like project.dataset.view
    parts = view_id.split(".")
    if len(parts) != 3:
        raise RuntimeError(f"Unexpected view_id format: {view_id}")
    dataset_view = f"{parts[1]}.{parts[2]}"
    # Delete if exists (older bq may not support --replace)
    rm_cmd = ["bq", f"--project_id={project}", "rm", "-f", "-t", dataset_view]
    try:
        subprocess.run(rm_cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass
    mk_cmd = [
        "bq",
        f"--project_id={project}",
        "mk",
        "--use_legacy_sql=false",
        f"--view={sql}",
        dataset_view,
    ]
    subprocess.run(mk_cmd, check=True)
    logger.info(f"[bq CLI] Created or replaced view {view_id}")


def create_or_replace_view(client: bigquery.Client, view_id: str, sql: str):
    view = bigquery.Table(view_id)
    view.view_query = sql
    try:
        try:
            client.delete_table(view_id, not_found_ok=True)
        except Exception:
            pass
        client.create_table(view)
        logger.info(f"Created view {view_id}")
    except Exception as e:
        msg = str(e)
        if any(s in msg for s in ("403", "Access Denied", "Forbidden")):
            # Fall back to bq CLI using the caller's gcloud auth
            project = os.environ.get("GOOGLE_CLOUD_PROJECT")
            _create_or_replace_view_via_cli(project, view_id, sql)
        else:
            raise


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=False, help="GCP project id (defaults to GOOGLE_CLOUD_PROJECT)")
    p.add_argument("--dataset", required=False, help="BQ dataset (defaults to BIGQUERY_TRAINING_DATASET)")
    p.add_argument("--dry_run", action="store_true", help="Print planned views and exit without touching BigQuery")
    args = p.parse_args()

    project = args.project or os.environ.get("GOOGLE_CLOUD_PROJECT")
    dataset = args.dataset or os.environ.get("BIGQUERY_TRAINING_DATASET")
    if not project or not dataset:
        if args.dry_run:
            print("[dry_run] Missing project/dataset; would create standard views if configured.")
            return
        raise RuntimeError(
            "Missing project/dataset. Set env GOOGLE_CLOUD_PROJECT and BIGQUERY_TRAINING_DATASET, "
            "or pass --project and --dataset."
        )
    # Use the same BigQuery client factory as other pipelines so env knobs
    # like AELP2_BQ_USE_GCE / AELP2_BQ_CREDENTIALS are respected.
    client = None
    if not args.dry_run:
        client = get_bq_client()
    dataset_ref = f"{project}.{dataset}"
    # Ensure dataset exists (safe for R&D datasets)
    try:
        client.get_dataset(f"{project}.{dataset}")
    except Exception:
        try:
            ds = bigquery.Dataset(f"{project}.{dataset}")
            if os.environ.get('BIGQUERY_DATASET_LOCATION'):
                ds.location = os.environ['BIGQUERY_DATASET_LOCATION']
            client.create_dataset(ds)
            logger.info(f"Created dataset: {dataset_ref}")
        except Exception as e:
            logger.warning(f"Could not create dataset {dataset_ref}: {e}")

    training_sql = f"""
        SELECT
          DATE(timestamp) AS date,
          COUNT(*) AS episode_rows,
          SUM(steps) AS steps,
          SUM(auctions) AS auctions,
          SUM(wins) AS wins,
          SUM(spend) AS spend,
          SUM(revenue) AS revenue,
          SUM(conversions) AS conversions,
          SAFE_DIVIDE(SUM(revenue), NULLIF(SUM(spend),0)) AS roas,
          SAFE_DIVIDE(SUM(spend), NULLIF(SUM(conversions),0)) AS cac,
          AVG(win_rate) AS avg_win_rate
        FROM `{dataset_ref}.training_episodes`
        GROUP BY date
        ORDER BY date
    """

    ads_sql = f"""
        WITH daily AS (
          SELECT 
            DATE(date) AS date,
            customer_id,
            campaign_id,
            MAX(impressions) AS impressions,
            MAX(clicks) AS clicks,
            MAX(cost_micros) AS cost_micros,
            MAX(conversions) AS conversions,
            MAX(conversion_value) AS conversion_value
          FROM `{dataset_ref}.ads_campaign_performance`
          GROUP BY date, customer_id, campaign_id
        )
        SELECT
          date,
          SUM(impressions) AS impressions,
          SUM(clicks) AS clicks,
          SUM(cost_micros)/1e6 AS cost,
          SUM(conversions) AS conversions,
          SUM(conversion_value) AS revenue,
          SAFE_DIVIDE(SUM(conversions), NULLIF(SUM(clicks),0)) AS cvr,
          SAFE_DIVIDE(SUM(clicks), NULLIF(SUM(impressions),0)) AS ctr,
          SAFE_DIVIDE(SUM(cost_micros)/1e6, NULLIF(SUM(conversions),0)) AS cac,
          SAFE_DIVIDE(SUM(conversion_value), NULLIF(SUM(cost_micros)/1e6,0)) AS roas
        FROM daily
        GROUP BY date
        ORDER BY date
    """

    for view_id, sql in [
        (f"{dataset_ref}.training_episodes_daily", training_sql),
        (f"{dataset_ref}.ads_campaign_daily", ads_sql),
    ]:
        try:
            create_or_replace_view(client, view_id, sql)
        except Exception as e:
            logger.warning(f"Skipping view {view_id}: {e}")

    # Meta Ads daily (if meta_ad_performance exists): date, cost, conversions, revenue
    try:
        if args.dry_run:
            raise Exception("skip in dry_run")
        client.get_table(f"{dataset_ref}.meta_ad_performance")
        meta_daily_sql = f"""
            SELECT
              DATE(date) AS date,
              SUM(cost) AS cost,
              SUM(conversions) AS conversions,
              SUM(revenue) AS revenue
            FROM `{dataset_ref}.meta_ad_performance`
            GROUP BY date
            ORDER BY date
        """
        try:
            create_or_replace_view(client, f"{dataset_ref}.meta_ads_daily", meta_daily_sql)
        except Exception as e:
            logger.warning(f"Skipping view {dataset_ref}.meta_ads_daily: {e}")
        # Per-campaign daily for Meta
        meta_campaign_daily_sql = f"""
            SELECT
              DATE(date) AS date,
              CAST(campaign_id AS STRING) AS campaign_id,
              SUM(cost) AS cost,
              SUM(conversions) AS conversions,
              SUM(revenue) AS revenue
            FROM `{dataset_ref}.meta_ad_performance`
            GROUP BY date, campaign_id
            ORDER BY date
        """
        try:
            create_or_replace_view(client, f"{dataset_ref}.meta_campaign_daily", meta_campaign_daily_sql)
        except Exception as e:
            logger.warning(f"Skipping view {dataset_ref}.meta_campaign_daily: {e}")
    except Exception:
        logger.info("meta_ad_performance table not found; skipping meta_ads_daily view")

    # Combined Ads daily (Google Ads + Meta) when both are present
    try:
        if args.dry_run:
            raise Exception("skip in dry_run")
        # Ensure base views exist
        client.get_table(f"{dataset_ref}.ads_campaign_performance")
        client.get_table(f"{dataset_ref}.meta_ad_performance")
        combined_sql = f"""
            WITH g AS (
              SELECT DATE(date) AS date,
                     SUM(cost_micros)/1e6 AS cost,
                     SUM(conversions) AS conversions,
                     SUM(conversion_value) AS revenue
              FROM `{dataset_ref}.ads_campaign_performance`
              GROUP BY date
            ), m AS (
              SELECT DATE(date) AS date,
                     SUM(cost) AS cost,
                     SUM(conversions) AS conversions,
                     SUM(revenue) AS revenue
              FROM `{dataset_ref}.meta_ad_performance`
              GROUP BY date
            )
            SELECT date,
                   SAFE_CAST(SUM(cost) AS FLOAT64) AS cost,
                   SAFE_CAST(SUM(conversions) AS FLOAT64) AS conversions,
                   SAFE_CAST(SUM(revenue) AS FLOAT64) AS revenue
            FROM (
              SELECT * FROM g
              UNION ALL
              SELECT * FROM m
            )
            GROUP BY date
            ORDER BY date
        """
        try:
            create_or_replace_view(client, f"{dataset_ref}.ads_all_daily", combined_sql)
        except Exception as e:
            logger.warning(f"Skipping view {dataset_ref}.ads_all_daily: {e}")
    except Exception:
        logger.info("Combined ads_all_daily skipped (one or both base tables missing)")

    # Subagents daily view: proposal counts by subagent and event_type
    try:
        if args.dry_run:
            raise Exception("skip in dry_run")
        client.get_table(f"{dataset_ref}.subagent_events")
        subagents_sql = f"""
            SELECT
              DATE(timestamp) AS date,
              subagent,
              event_type,
              COUNT(*) AS events
            FROM `{dataset_ref}.subagent_events`
            GROUP BY date, subagent, event_type
            ORDER BY date
        """
        try:
            create_or_replace_view(client, f"{dataset_ref}.subagents_daily", subagents_sql)
        except Exception as e:
            logger.warning(f"Skipping view {dataset_ref}.subagents_daily: {e}")
    except Exception:
        logger.info("subagent_events table not found; skipping subagents_daily view")

    # Bidding events per-minute aggregate view (if bidding_events exists)
    try:
        if args.dry_run:
            raise Exception("skip in dry_run")
        client.get_table(f"{dataset_ref}.bidding_events")
        bidding_sql = f"""
            SELECT
              TIMESTAMP_TRUNC(timestamp, MINUTE) AS minute,
              COUNT(*) AS auctions,
              COUNTIF(won) AS wins,
              SAFE_DIVIDE(COUNTIF(won), NULLIF(COUNT(*),0)) AS win_rate,
              AVG(bid_amount) AS avg_bid,
              AVG(price_paid) AS avg_price_paid
            FROM `{dataset_ref}.bidding_events`
            WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
            GROUP BY minute
            ORDER BY minute
        """
        try:
            create_or_replace_view(client, f"{dataset_ref}.bidding_events_per_minute", bidding_sql)
        except Exception as e:
            logger.warning(f"Skipping view {dataset_ref}.bidding_events_per_minute: {e}")
    except Exception:
        logger.info("bidding_events table not found; skipping per-minute view")

    # Optional GA4 views
    # 1) Aggregates table in training dataset
    try:
        if args.dry_run:
            raise Exception("skip in dry_run")
        client.get_table(f"{dataset_ref}.ga4_aggregates")
        ga4_sql = f"""
            SELECT
              date AS date,
              default_channel_group,
              device_category,
              SUM(sessions) AS sessions,
              SUM(conversions) AS conversions,
              SUM(users) AS users
            FROM `{dataset_ref}.ga4_aggregates`
            GROUP BY date, default_channel_group, device_category
            ORDER BY date
        """
        try:
            create_or_replace_view(client, f"{dataset_ref}.ga4_daily", ga4_sql)
        except Exception as e:
            logger.warning(f"Skipping view {dataset_ref}.ga4_daily: {e}")
    except Exception:
        logger.info("ga4_aggregates table not found; skipping GA4 aggregates view")

    # Optional GA4 lagged attribution view
    try:
        if args.dry_run:
            raise Exception("skip in dry_run")
        client.get_table(f"{dataset_ref}.ga4_lagged_attribution")
        ga4_lagged_sql = f"""
            SELECT date, SUM(ga4_conversions_lagged) AS ga4_conversions_lagged
            FROM `{dataset_ref}.ga4_lagged_attribution`
            GROUP BY date
            ORDER BY date
        """
        try:
            create_or_replace_view(client, f"{dataset_ref}.ga4_lagged_daily", ga4_lagged_sql)
        except Exception as e:
            logger.warning(f"Skipping view {dataset_ref}.ga4_lagged_daily: {e}")
    except Exception:
        logger.info("ga4_lagged_attribution table not found; skipping GA4 lagged view")

    # 2) Native GA4 export dataset (optional): set GA4_EXPORT_DATASET env var
    ga4_export_dataset = os.environ.get("GA4_EXPORT_DATASET")
    if ga4_export_dataset:
        try:
            # Create a staging daily view from export events_*
            # Support cross-project: GA4_EXPORT_DATASET may be "dataset" or "project.dataset"
            if "." in ga4_export_dataset:
                export_ref = ga4_export_dataset  # already qualified
            else:
                export_ref = f"{project}.{ga4_export_dataset}"
            # Check if any partitioned events tables exist
            if args.dry_run:
                raise Exception("skip in dry_run")
            # list_tables expects dataset reference; ensure project and dataset split
            try:
                exp_proj, exp_ds = export_ref.split(".", 1)
            except ValueError:
                exp_proj, exp_ds = project, ga4_export_dataset
            _ = list(client.list_tables(f"{exp_proj}.{exp_ds}"))
            ga4_export_sql = f"""
                SELECT
                  event_date AS date,
                  device.category AS device_category,
                  traffic_source.medium AS medium,
                  traffic_source.name AS source,
                  COUNTIF(event_name = 'session_start') AS sessions,
                  COUNTIF(event_name = 'purchase') AS conversions,
                  COUNT(*) AS events
                FROM `{export_ref}.events_*`
                WHERE _TABLE_SUFFIX BETWEEN FORMAT_DATE('%Y%m%d', DATE_SUB(CURRENT_DATE(), INTERVAL 90 DAY))
                                      AND FORMAT_DATE('%Y%m%d', CURRENT_DATE())
                GROUP BY date, device_category, medium, source
                ORDER BY date
            """
            try:
                create_or_replace_view(client, f"{dataset_ref}.ga4_export_daily", ga4_export_sql)
            except Exception as e:
                logger.warning(f"Skipping view {dataset_ref}.ga4_export_daily: {e}")
        except Exception as e:
            logger.info(f"GA4 export dataset not accessible or no events tables: {e}")

    # Optional KPI-only Ads daily view using conversion_action_stats
    kpi_ids = os.environ.get("AELP2_KPI_CONVERSION_ACTION_IDS")
    if kpi_ids:
        ids = ",".join([f"'{x.strip()}'" for x in kpi_ids.split(',') if x.strip()])
        if ids:
            # Check whether ads_conversion_actions has include_in_conversions_metric; older datasets may lack it
            include_filter_sql = ""
            try:
                if args.dry_run:
                    raise Exception("skip in dry_run")
                tbl = client.get_table(f"{dataset_ref}.ads_conversion_actions")
                field_names = {f.name for f in tbl.schema}
                if "include_in_conversions_metric" in field_names:
                    include_filter_sql = "AND (a.include_in_conversions_metric IS NULL OR a.include_in_conversions_metric = TRUE)"
                else:
                    include_filter_sql = ""  # Skip filter if field missing
            except Exception:
                include_filter_sql = ""

            ads_kpi_sql = f"""
                WITH kpi AS (
                  SELECT DATE(s.date) AS date,
                         SUM(s.conversions) AS conversions,
                         SUM(s.conversion_value) AS revenue
                  FROM `{dataset_ref}.ads_conversion_action_stats` s
                  LEFT JOIN `{dataset_ref}.ads_conversion_actions` a
                    ON a.conversion_action_id = s.conversion_action_id
                  WHERE s.conversion_action_id IN ({ids})
                    {include_filter_sql}
                  GROUP BY date
                ),
                cost AS (
                  SELECT DATE(date) AS date,
                         SUM(cost_micros)/1e6 AS cost
                  FROM `{dataset_ref}.ads_campaign_performance`
                  GROUP BY date
                )
                SELECT kpi.date,
                       kpi.conversions,
                       kpi.revenue,
                       cost.cost,
                       SAFE_DIVIDE(cost.cost, NULLIF(kpi.conversions,0)) AS cac,
                       SAFE_DIVIDE(kpi.revenue, NULLIF(cost.cost,0)) AS roas
                FROM kpi
                LEFT JOIN cost USING(date)
                ORDER BY kpi.date
            """
            if args.dry_run:
                print(f"[dry_run] Would create/replace view {dataset_ref}.ads_kpi_daily with KPI-only CAC/ROAS.")
            else:
                try:
                    create_or_replace_view(client, f"{dataset_ref}.ads_kpi_daily", ads_kpi_sql)
                except Exception as e:
                    logger.warning(f"Skipping view {dataset_ref}.ads_kpi_daily: {e}")

    # AB experiments daily view (if tables exist)
    try:
        if args.dry_run:
            raise Exception("skip in dry_run")
        client.get_table(f"{dataset_ref}.ab_experiments")
        sql = f"""
            SELECT
              DATE(COALESCE(end, start)) AS date,
              COUNT(*) AS experiments,
              COUNTIF(status='running') AS running,
              COUNTIF(status='proposed') AS proposed
            FROM `{dataset_ref}.ab_experiments`
            GROUP BY date
            ORDER BY date
        """
        try:
            create_or_replace_view(client, f"{dataset_ref}.ab_experiments_daily", sql)
        except Exception as e:
            logger.warning(f"Skipping view {dataset_ref}.ab_experiments_daily: {e}")
    except Exception:
        logger.info("ab_experiments table not found; skipping daily view")

    # AB exposures daily view (if table exists)
    try:
        if args.dry_run:
            raise Exception("skip in dry_run")
        client.get_table(f"{dataset_ref}.ab_exposures")
        sql = f"""
            SELECT DATE(timestamp) AS date, COUNT(*) AS exposures
            FROM `{dataset_ref}.ab_exposures`
            GROUP BY date
            ORDER BY date
        """
        try:
            create_or_replace_view(client, f"{dataset_ref}.ab_exposures_daily", sql)
        except Exception as e:
            logger.warning(f"Skipping view {dataset_ref}.ab_exposures_daily: {e}")
    except Exception:
        logger.info("ab_exposures table not found; skipping daily view")

    if args.dry_run:
        print("[dry_run] Planned views: training_episodes_daily, ads_campaign_daily, optional: subagents_daily, bidding_events_per_minute, ga4_daily, ga4_lagged_daily, ga4_export_daily, ads_kpi_daily (if KPI IDs set), ab_experiments_daily (if table), ab_exposures_daily (if table).")
        return

    # Impact partner daily + platform daily
    try:
        # Try to use impact_partner_performance if present; otherwise derive from invoices
        use_perf = False
        try:
            client.get_table(f"{dataset_ref}.impact_partner_performance")
            use_perf = True
        except Exception:
            use_perf = False

        if use_perf:
            # Prefer performance payout when available (>0); otherwise fall back to invoices for that day+partner.
            # Also add any external ACH costs if provided (table: affiliates_external_costs).
            sql_partner = f"""
                WITH perf AS (
                  SELECT DATE(date) AS date,
                         SAFE_CAST(partner_id AS STRING) AS partner_id,
                         partner,
                         SUM(actions) AS actions,
                         SUM(payout) AS payout,
                         SUM(revenue) AS revenue
                  FROM `{dataset_ref}.impact_partner_performance`
                  GROUP BY date, partner_id, partner
                ), inv AS (
                  SELECT DATE(CreatedDate) AS date,
                         SAFE_CAST(MediaId AS STRING) AS partner_id,
                         ANY_VALUE(MediaName) AS partner,
                         COALESCE(
                           SUM(IFNULL(dli.ActionAmount,0)),
                           SUM(IFNULL(dli.OtherAmount,0)),
                           SUM(IFNULL(inv.TotalAmount,0))
                         ) AS payout_inv
                  FROM `{dataset_ref}.impact_invoices` inv,
                       UNNEST(IFNULL(inv.DetailedLineItems, [])) AS dli
                  GROUP BY date, partner_id
                ), ext AS (
                  SELECT DATE(date) AS date,
                         SAFE_CAST(partner_id AS STRING) AS partner_id,
                         SUM(amount) AS payout_ext
                  FROM `{dataset_ref}.affiliates_external_costs`
                  GROUP BY date, partner_id
                ), merged AS (
                  SELECT p.date, p.partner_id, p.partner,
                         IFNULL(p.actions, 0.0) AS actions,
                         (CASE WHEN IFNULL(p.payout,0.0) > 0.0 THEN p.payout ELSE IFNULL(i.payout_inv, 0.0) END) + IFNULL(e.payout_ext,0.0) AS payout,
                         p.revenue
                  FROM perf p
                  LEFT JOIN inv i ON i.date = p.date AND i.partner_id = p.partner_id
                  LEFT JOIN ext e ON e.date = p.date AND e.partner_id = p.partner_id
                  UNION ALL
                  -- Invoice-only rows where no perf record exists for that day+partner
                  SELECT i.date, i.partner_id, IFNULL(i.partner, '(unknown)') AS partner,
                         0.0 AS actions,
                         i.payout_inv + IFNULL(e.payout_ext,0.0) AS payout,
                         NULL AS revenue
                  FROM inv i
                  LEFT JOIN ext e ON e.date = i.date AND e.partner_id = i.partner_id
                  WHERE NOT EXISTS (
                    SELECT 1 FROM perf p
                    WHERE p.date = i.date AND p.partner_id = i.partner_id
                  )
                )
                SELECT * FROM merged
                ORDER BY date DESC
            """
        else:
            # Fallback: derive daily payout from invoices using CreatedDate and DetailedLineItems
            sql_partner = f"""
                WITH base AS (
                  SELECT DATE(CreatedDate) AS date,
                         SAFE_CAST(MediaId AS STRING) AS partner_id,
                         MediaName AS partner,
                         0.0 AS actions,
                         -- Prefer ActionAmount if present; else OtherAmount; else TotalAmount
                         COALESCE(
                           SUM(IFNULL(dli.ActionAmount,0)),
                           SUM(IFNULL(dli.OtherAmount,0)),
                           SUM(IFNULL(inv.TotalAmount,0))
                         ) AS payout,
                         NULL AS revenue
                  FROM `{dataset_ref}.impact_invoices` inv,
                       UNNEST(IFNULL(inv.DetailedLineItems, [])) AS dli
                  GROUP BY date, partner_id, partner
                )
                SELECT * FROM base
                ORDER BY date DESC
            """
        create_or_replace_view(client, f"{dataset_ref}.impact_partner_daily", sql_partner)

        sql_platform = f"""
            WITH base AS (
              SELECT date,
                     SUM(payout) AS partner_cost,
                     SUM(actions) AS actions,
                     SUM(revenue) AS revenue
              FROM `{dataset_ref}.impact_partner_daily`
              GROUP BY date
            ), ext AS (
              -- Add external ACH rows that are not mapped to a specific partner (partner_id IS NULL)
              SELECT DATE(date) AS date, SUM(amount) AS ext_amount
              FROM `{dataset_ref}.affiliates_external_costs`
              WHERE partner_id IS NULL
              GROUP BY date
            )
            SELECT b.date,
                   (b.partner_cost + IFNULL(e.ext_amount, 0)) AS cost,
                   b.actions,
                   b.revenue
            FROM base b
            LEFT JOIN ext e USING(date)
            ORDER BY b.date
        """
        create_or_replace_view(client, f"{dataset_ref}.impact_platform_daily", sql_platform)
    except Exception as e:
        logger.warning(f"Skipping Impact views: {e}")

    # Cross-platform KPI daily view (Ads cost + GA conversions)
    try:
        sql = f"""
        WITH ads AS (
          -- Google Ads daily (already rolled up)
          SELECT DATE(date) AS date,
                 'google_ads' AS platform,
                 'paid_search' AS channel_group,
                 SUM(cost) AS cost,
                 SUM(revenue) AS revenue
          FROM `{dataset_ref}.ads_campaign_daily`
          GROUP BY date, platform, channel_group
          UNION ALL
          -- Impact.com platform daily (cost = payout)
          SELECT date AS date,
                 'impact' AS platform,
                 'affiliate' AS channel_group,
                 cost AS cost,
                 revenue AS revenue
          FROM `{dataset_ref}.impact_platform_daily`
        ), ga AS (
          SELECT date,
                 'ga4' AS platform,
                 'all' AS channel_group,
                 NULL AS cost,
                 NULL AS revenue,
                 (SELECT SUM(enrollments)
                    FROM `{dataset_ref}.ga4_derived_daily` gd
                   WHERE gd.date = d.date) AS conversions
          FROM (SELECT DISTINCT date FROM `{dataset_ref}.ga4_derived_daily`) d
        )
        SELECT a.date,
               a.platform,
               a.channel_group,
               a.cost,
               (SELECT SUM(enrollments)
                  FROM `{dataset_ref}.ga4_derived_daily` gd
                 WHERE gd.date = a.date) AS conversions,
               a.revenue
        FROM ads a
        UNION ALL
        SELECT date, platform, channel_group, cost, conversions, revenue FROM ga
        ORDER BY date, platform
        """
        create_or_replace_view(client, f"{dataset_ref}.cross_platform_kpi_daily", sql)
    except Exception as e:
        logger.warning(f"Skipping view {dataset_ref}.cross_platform_kpi_daily: {e}")

    # Blended paid KPI daily (Google Ads + Impact cost; GA conversions)
    try:
        sql = f"""
        WITH costs AS (
          SELECT DATE(date) AS date,
                 SUM(CASE WHEN platform='google_ads' THEN cost ELSE 0 END) AS google_cost,
                 SUM(CASE WHEN platform='impact' THEN cost ELSE 0 END) AS impact_cost
          FROM `{dataset_ref}.cross_platform_kpi_daily`
          GROUP BY date
        ), conv AS (
          SELECT DATE(date) AS date,
                 SUM(enrollments) AS conversions
          FROM `{dataset_ref}.ga4_derived_daily`
          GROUP BY date
        )
        SELECT c.date,
               (google_cost + impact_cost) AS cost,
               conv.conversions AS conversions,
               NULL AS revenue
        FROM costs c
        LEFT JOIN conv USING(date)
        ORDER BY c.date
        """
        create_or_replace_view(client, f"{dataset_ref}.blended_paid_kpi_daily", sql)
    except Exception as e:
        logger.warning(f"Skipping view {dataset_ref}.blended_paid_kpi_daily: {e}")

    # Impact KPI daily (using GA enrollments as conversions placeholder)
    try:
        sql = f"""
        SELECT p.date AS date,
               p.cost AS cost,
               (SELECT SUM(enrollments) FROM `{dataset_ref}.ga4_derived_daily` g WHERE g.date = p.date) AS conversions,
               NULL AS revenue
        FROM `{dataset_ref}.impact_platform_daily` p
        ORDER BY date
        """
        create_or_replace_view(client, f"{dataset_ref}.impact_kpi_daily", sql)
    except Exception as e:
        logger.warning(f"Skipping view {dataset_ref}.impact_kpi_daily: {e}")

    # Impact Partner KPI daily (cost = payout by partner; conversions = affiliate-triggered by partner)
    try:
        sql = f"""
        WITH c AS (
          SELECT date, partner_id, triggered_conversions
          FROM `{dataset_ref}.ga_affiliate_triggered_by_partner_daily`
        )
        SELECT p.date, p.partner_id, p.partner,
               p.payout AS cost,
               c.triggered_conversions AS conversions,
               NULL AS revenue
        FROM `{dataset_ref}.impact_partner_daily` p
        LEFT JOIN c USING(date, partner_id)
        ORDER BY date
        """
        create_or_replace_view(client, f"{dataset_ref}.impact_partner_kpi_daily", sql)
    except Exception as e:
        logger.warning(f"Skipping view {dataset_ref}.impact_partner_kpi_daily: {e}")

    # Impact seed→harvest correlation stats (lag 0..7 days)
    try:
        sql = f"""
        WITH base AS (
          SELECT p.date, p.cost,
                 (SELECT SUM(enrollments) FROM `{dataset_ref}.ga4_derived_daily` g WHERE g.date = p.date) AS enrollments
          FROM `{dataset_ref}.impact_platform_daily` p
        ), lags AS (
          SELECT b1.date AS date, b1.cost, b2.enrollments AS enrollments_lead,
                 lag_val AS lag
          FROM base b1
          CROSS JOIN UNNEST([0,1,2,3,4,5,6,7]) AS lag_val
          JOIN base b2
          ON b2.date = DATE_ADD(b1.date, INTERVAL lag_val DAY)
        )
        SELECT lag, CORR(cost, enrollments_lead) AS corr
        FROM lags
        GROUP BY lag
        ORDER BY lag
        """
        create_or_replace_view(client, f"{dataset_ref}.impact_seed_harvest_stats", sql)
    except Exception as e:
        logger.warning(f"Skipping view {dataset_ref}.impact_seed_harvest_stats: {e}")

    # Affiliate clickID joins: partner influence by day (user-level join via click_id)
    try:
        sql = f"""
        WITH clicks AS (
          SELECT DATE(date) AS date, click_id, SAFE_CAST(partner_id AS STRING) AS partner_id, ANY_VALUE(partner) AS partner
          FROM `{dataset_ref}.impact_clicks`
          GROUP BY date, click_id, partner_id
        ), ga_clicks AS (
          SELECT DATE(date) AS date, user_pseudo_id, click_id, event_ts
          FROM `{dataset_ref}.ga_affiliate_clickids`
        ), purchases AS (
          SELECT DATE(TIMESTAMP_MICROS(event_timestamp)) AS date,
                 user_pseudo_id,
                 TIMESTAMP_MICROS(event_timestamp) AS p_ts
          FROM `{export_ds}.events_*`
          WHERE _TABLE_SUFFIX BETWEEN FORMAT_DATE('%Y%m%d', DATE_SUB(CURRENT_DATE(), INTERVAL 120 DAY))
                                  AND FORMAT_DATE('%Y%m%d', CURRENT_DATE())
            AND event_name='purchase'
        ), joined AS (
          SELECT p.date, p.user_pseudo_id, p.p_ts, g.click_id, g.event_ts AS c_ts
          FROM purchases p
          JOIN ga_clicks g USING(user_pseudo_id)
          WHERE g.event_ts <= p.p_ts AND TIMESTAMP_DIFF(p.p_ts, g.event_ts, DAY) BETWEEN 0 AND 14
        ), picked AS (
          SELECT date, user_pseudo_id, p_ts, click_id,
                 ROW_NUMBER() OVER (PARTITION BY user_pseudo_id, p_ts ORDER BY c_ts DESC) AS rn,
                 TIMESTAMP_DIFF(p_ts, c_ts, DAY) AS lag_days
          FROM joined
        ), with_partner AS (
          SELECT pk.date, pk.user_pseudo_id, pk.p_ts, pk.lag_days,
                 c.partner_id, c.partner
          FROM picked pk
          LEFT JOIN clicks c USING(click_id)
          WHERE pk.rn=1
        )
        SELECT date, partner_id, ANY_VALUE(partner) AS partner, COUNT(*) AS attributable_enrollments,
               APPROX_QUANTILES(lag_days, 3)[OFFSET(2)] AS median_lag_days
        FROM with_partner
        GROUP BY date, partner_id
        ORDER BY date DESC, attributable_enrollments DESC
        """
        create_or_replace_view(client, f"{dataset_ref}.affiliate_partner_influence_daily", sql)
    except Exception as e:
        logger.warning(f"Skipping view {dataset_ref}.affiliate_partner_influence_daily: {e}")

    # KPI (Pacer-aligned) daily view — source of truth for CAC
    # Uses pacing_pacer_daily (from Excel) to compute CAC = spend / d2c_total_subscribers
    try:
        sql = f"""
        WITH pacer AS (
          SELECT date,
                 SUM(spend) AS spend,
                 SUM(d2c_total_subscribers) AS subscribers
          FROM `{dataset_ref}.pacing_pacer_daily`
          GROUP BY date
        )
        SELECT date,
               spend,
               subscribers,
               SAFE_DIVIDE(spend, NULLIF(subscribers,0)) AS cac
        FROM pacer
        ORDER BY date
        """
        create_or_replace_view(client, f"{dataset_ref}.kpi_pacer_daily", sql)
    except Exception as e:
        logger.warning(f"Skipping view {dataset_ref}.kpi_pacer_daily: {e}")

    # GA-aligned KPI view: conversions = aligned_enrollments (GA + non-GA delta), cost = pacer spend
    try:
        sql = f"""
        SELECT date,
               SAFE_CAST(spend AS FLOAT64) AS cost,
               SAFE_CAST(aligned_enrollments AS FLOAT64) AS conversions,
               NULL AS revenue,
               SAFE_DIVIDE(SAFE_CAST(spend AS FLOAT64), NULLIF(SAFE_CAST(aligned_enrollments AS FLOAT64),0)) AS cac
        FROM `{dataset_ref}.ga_aligned_daily`
        ORDER BY date
        """
        create_or_replace_view(client, f"{dataset_ref}.aligned_kpi_daily", sql)
    except Exception as e:
        logger.warning(f"Skipping view {dataset_ref}.aligned_kpi_daily: {e}")

    # MMM covariates daily view: joins aligned KPI with Ads capacity and GA demand proxies
    try:
        sql = f"""
        WITH ads AS (
          SELECT DATE(date) AS date,
                 SUM(cost_micros)/1e6 AS ads_cost,
                 AVG(impression_share) AS is_avg,
                 AVG(lost_is_budget) AS lost_is_budget,
                 AVG(lost_is_rank) AS lost_is_rank,
                 AVG(top_impression_share) AS top_is,
                 AVG(abs_top_impression_share) AS abs_top_is,
                 -- brand share heuristic (only if names not redacted)
                 SAFE_DIVIDE(SUM(IF(LOWER(campaign_name) LIKE '%aura%' OR LOWER(campaign_name) LIKE '%brand%', cost_micros/1e6, NULL)), NULLIF(SUM(cost_micros)/1e6,0)) AS brand_cost_share
          FROM `{dataset_ref}.ads_campaign_performance`
          GROUP BY date
        ),
        hi AS (
          SELECT date,
                 SUM(begin_checkout) AS hi_begin_checkout,
                 SUM(form_submit_enroll) AS hi_form_submit_enroll,
                 SUM(high_intent_no_purchase_7) AS hi_no_purchase_7
          FROM `{dataset_ref}.ga4_high_intent_daily`
          GROUP BY date
        ),
        aff AS (
          SELECT date, SUM(triggered_conversions) AS affiliate_triggered
          FROM `{dataset_ref}.ga_affiliate_triggered_daily`
          GROUP BY date
        ),
        promo AS (
          SELECT date,
                 SAFE_DIVIDE(SUM(CASE WHEN REGEXP_CONTAINS(LOWER(CONCAT(IFNULL(plan_code,''), '|', IFNULL(cc,''))), r'promo|discount|intro|ft|free|bestoffer|off[0-9]+') THEN purchases END), NULLIF(SUM(purchases),0)) AS promo_share
          FROM `{dataset_ref}.ga4_offer_code_daily`
          GROUP BY date
        ),
        promo_cal AS (
          SELECT date, promo_flag, promo_intensity FROM `{dataset_ref}.promo_calendar`
        )
        SELECT a.date,
               k.cost,
               k.conversions,
               k.cac,
               ads.ads_cost,
               ads.is_avg,
               ads.lost_is_budget,
               ads.lost_is_rank,
               ads.top_is,
               ads.abs_top_is,
               ads.brand_cost_share,
               hi.hi_begin_checkout,
               hi.hi_form_submit_enroll,
               hi.hi_no_purchase_7,
               promo.promo_share,
               aff.affiliate_triggered,
               promo_cal.promo_flag,
               promo_cal.promo_intensity,
               EXTRACT(DAYOFWEEK FROM a.date) AS dow,
               EXTRACT(MONTH FROM a.date) AS month,
               EXTRACT(DAY FROM a.date) AS dom,
               IF(EXTRACT(DAYOFWEEK FROM a.date) IN (1,7), 1, 0) AS is_weekend
        FROM `{dataset_ref}.aligned_kpi_daily` k
        JOIN (SELECT DISTINCT date FROM `{dataset_ref}.ga_aligned_daily`) a USING(date)
        LEFT JOIN ads USING(date)
        LEFT JOIN hi USING(date)
        LEFT JOIN promo USING(date)
        LEFT JOIN aff USING(date)
        LEFT JOIN promo_cal USING(date)
        ORDER BY a.date
        """
        create_or_replace_view(client, f"{dataset_ref}.mmm_covariates_daily", sql)
    except Exception as e:
        logger.warning(f"Skipping view {dataset_ref}.mmm_covariates_daily: {e}")

    # Triggered (touch-aligned) KPI view: conversions from ga_triggered_daily; cost = pacer spend
    try:
        sql = f"""
        WITH pacer AS (
          SELECT date, SUM(spend) AS spend
          FROM `{dataset_ref}.pacing_pacer_daily`
          GROUP BY date
        )
        SELECT t.date,
               p.spend AS cost,
               SAFE_CAST(t.triggered_conversions AS FLOAT64) AS conversions,
               NULL AS revenue,
               SAFE_DIVIDE(p.spend, NULLIF(SAFE_CAST(t.triggered_conversions AS FLOAT64),0)) AS cac
        FROM `{dataset_ref}.ga_triggered_daily` t
        LEFT JOIN pacer p USING(date)
        ORDER BY t.date
        """
        create_or_replace_view(client, f"{dataset_ref}.triggered_kpi_daily", sql)
    except Exception as e:
        logger.warning(f"Skipping view {dataset_ref}.triggered_kpi_daily: {e}")

    # Affiliate-triggered by partner: join GA source with Impact partner domains (heuristic)
    try:
        sql = f"""
        WITH src_daily AS (
          SELECT date,
                 SPLIT(channel,'/')[OFFSET(0)] AS source,
                 SUM(triggered_conversions) AS triggered
          FROM `{dataset_ref}.ga_affiliate_triggered_channel_daily`
          GROUP BY date, source
        ), dom AS (
          SELECT DISTINCT SAFE_CAST(partner_id AS STRING) AS partner_id, partner, root
          FROM `{dataset_ref}.impact_partner_domains`
        )
        SELECT sd.date, d.partner_id, d.partner,
               SUM(sd.triggered) AS triggered_conversions
        FROM src_daily sd
        JOIN dom d
          ON REGEXP_CONTAINS(LOWER(sd.source), r'(^|[^a-z0-9])' || LOWER(d.root) || r'([^a-z0-9]|$)')
        GROUP BY date, partner_id, partner
        ORDER BY date DESC, triggered_conversions DESC
        """
        create_or_replace_view(client, f"{dataset_ref}.ga_affiliate_triggered_by_partner_daily", sql)
    except Exception as e:
        logger.warning(f"Skipping view {dataset_ref}.ga_affiliate_triggered_by_partner_daily: {e}")

    # Affiliate brand-lift (corr vs brand search conversions at best lag)
    try:
        sql = f"""
        WITH brand AS (
          SELECT DATE(date) AS date, SUM(conversions) AS brand_conv
          FROM `{dataset_ref}.search_brand_daily`
          GROUP BY date
        ), P AS (
          SELECT date, partner_id, partner, triggered_conversions FROM `{dataset_ref}.ga_affiliate_triggered_by_partner_daily`
        ), L AS (
          SELECT p1.partner_id, ANY_VALUE(p1.partner) AS partner, lag_val AS lag,
                 COUNT(*) AS n_days,
                 CORR(p1.triggered_conversions, b2.brand_conv) AS corr
          FROM P p1
          JOIN brand b1 USING(date)
          CROSS JOIN UNNEST([0,1,2,3,4,5,6,7]) AS lag_val
          JOIN brand b2 ON b2.date = DATE_ADD(p1.date, INTERVAL lag_val DAY)
          GROUP BY partner_id, lag
        )
        SELECT partner_id, ANY_VALUE(partner) AS partner,
               ARRAY_AGG(STRUCT(lag, corr, n_days) ORDER BY ABS(corr) DESC LIMIT 1)[OFFSET(0)] AS best
        FROM L
        GROUP BY partner_id
        """
        create_or_replace_view(client, f"{dataset_ref}.affiliate_brand_lift_by_partner", sql)
    except Exception as e:
        logger.warning(f"Skipping view {dataset_ref}.affiliate_brand_lift_by_partner: {e}")

    # Search brand vs non-brand (Google Ads only) — directional (uses Ads conversions)
    try:
        brand_cond = "LOWER(campaign_name) LIKE '%brand%' OR LOWER(campaign_name) LIKE '%aura%'"
        search_brand_sql = f"""
            SELECT DATE(date) AS date,
                   SUM(IF({brand_cond}, cost_micros, 0))/1e6 AS cost,
                   SUM(IF({brand_cond}, conversions, 0)) AS conversions,
                   SUM(IF({brand_cond}, conversion_value, 0)) AS revenue
            FROM `{dataset_ref}.ads_campaign_performance`
            GROUP BY date
            ORDER BY date
        """
        search_nonbrand_sql = f"""
            SELECT DATE(date) AS date,
                   SUM(IF({brand_cond}, 0, cost_micros))/1e6 AS cost,
                   SUM(IF({brand_cond}, 0, conversions)) AS conversions,
                   SUM(IF({brand_cond}, 0, conversion_value)) AS revenue
            FROM `{dataset_ref}.ads_campaign_performance`
            GROUP BY date
            ORDER BY date
        """
        create_or_replace_view(client, f"{dataset_ref}.search_brand_daily", search_brand_sql)
        create_or_replace_view(client, f"{dataset_ref}.search_nonbrand_daily", search_nonbrand_sql)
    except Exception as e:
        logger.warning(f"Skipping view {dataset_ref}.search_brand_daily or nonbrand: {e}")

    # Aligned brand/nonbrand KPI (approximate split using brand_cost_share per day)
    try:
        sql_b = f"""
          WITH split AS (
            SELECT c.date,
                   SAFE_CAST(k.cost * c.brand_cost_share AS FLOAT64) AS cost,
                   SAFE_CAST(k.conversions * c.brand_cost_share AS FLOAT64) AS conversions
            FROM `{dataset_ref}.aligned_kpi_daily` k
            JOIN `{dataset_ref}.mmm_covariates_daily` c USING(date)
          )
          SELECT date, cost, conversions,
                 SAFE_DIVIDE(cost, NULLIF(conversions,0)) AS cac,
                 NULL AS revenue
          FROM split
          ORDER BY date
        """
        sql_nb = f"""
          WITH split AS (
            SELECT c.date,
                   SAFE_CAST(k.cost * (1.0 - c.brand_cost_share) AS FLOAT64) AS cost,
                   SAFE_CAST(k.conversions * (1.0 - c.brand_cost_share) AS FLOAT64) AS conversions
            FROM `{dataset_ref}.aligned_kpi_daily` k
            JOIN `{dataset_ref}.mmm_covariates_daily` c USING(date)
          )
          SELECT date, cost, conversions,
                 SAFE_DIVIDE(cost, NULLIF(conversions,0)) AS cac,
                 NULL AS revenue
          FROM split
          ORDER BY date
        """
        create_or_replace_view(client, f"{dataset_ref}.aligned_brand_kpi_daily", sql_b)
        create_or_replace_view(client, f"{dataset_ref}.aligned_nonbrand_kpi_daily", sql_nb)
    except Exception as e:
        logger.warning(f"Skipping aligned brand/nonbrand KPI views: {e}")


if __name__ == "__main__":
    main()
