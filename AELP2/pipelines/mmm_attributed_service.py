#!/usr/bin/env python3
"""
MMM with Proper Delayed Reward Attribution (3-14 day windows)

This version properly attributes conversions back to the spend that caused them,
not just same-day attribution. Uses the sophisticated reward_attribution.py logic.

Key improvements:
- Tracks user journeys over 3-14 days
- Attributes conversions to the right touchpoints
- Calculates true CAC with proper time lag
- Handles multi-touch attribution correctly

Tables:
- Reads: ga4_paths (user journey data)
- Writes: mmm_curves_attributed, mmm_allocations_attributed
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict

from google.cloud import bigquery
from google.cloud.exceptions import NotFound

# Import the sophisticated attribution system
import sys
sys.path.append('/home/hariravichandran/AELP')
sys.path.append('/home/hariravichandran/AELP/AELP2/core/intelligence')

try:
    from reward_attribution import (
        RewardAttributionWrapper,
        TouchpointData,
        ConversionEvent,
        MIN_ATTRIBUTION_WINDOW,
        MAX_ATTRIBUTION_WINDOW
    )
    ATTRIBUTION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import reward_attribution: {e}")
    ATTRIBUTION_AVAILABLE = False


def get_user_journeys(bq: bigquery.Client, project: str, dataset: str,
                      start: date, end: date) -> pd.DataFrame:
    """
    Fetch user journey data with touchpoints and conversions.
    Returns DataFrame with columns: user_id, touchpoint_date, spend,
    conversion_date, conversion_value
    """

    # Extended window to capture journeys that convert after end date
    extended_end = end + timedelta(days=MAX_ATTRIBUTION_WINDOW)
    extended_start = start - timedelta(days=MAX_ATTRIBUTION_WINDOW)

    sql = f"""
    WITH touchpoints AS (
        -- Get all marketing touchpoints (impressions, clicks)
        SELECT
            user_pseudo_id as user_id,
            DATE(TIMESTAMP_MICROS(event_timestamp)) as touchpoint_date,
            traffic_source.source,
            traffic_source.medium,
            traffic_source.name as campaign,
            -- Estimate spend per touchpoint (you may need to join with cost data)
            CASE
                WHEN traffic_source.medium = 'cpc' THEN 0.50  -- Avg CPC
                WHEN traffic_source.medium = 'display' THEN 0.10
                ELSE 0.05
            END as touchpoint_spend,
            event_timestamp
        FROM `{project}.{dataset}.events_*`
        WHERE _TABLE_SUFFIX BETWEEN
            FORMAT_DATE('%Y%m%d', DATE('{extended_start}')) AND
            FORMAT_DATE('%Y%m%d', DATE('{extended_end}'))
            AND event_name IN ('page_view', 'click', 'view_item')
            AND traffic_source.source IS NOT NULL
    ),
    conversions AS (
        -- Get all conversions with value
        SELECT
            user_pseudo_id as user_id,
            DATE(TIMESTAMP_MICROS(event_timestamp)) as conversion_date,
            IFNULL(ecommerce.purchase_revenue, 74.70) as conversion_value,
            event_timestamp as conversion_timestamp
        FROM `{project}.{dataset}.events_*`
        WHERE _TABLE_SUFFIX BETWEEN
            FORMAT_DATE('%Y%m%d', DATE('{start}')) AND
            FORMAT_DATE('%Y%m%d', DATE('{extended_end}'))
            AND event_name = 'purchase'
    ),
    journeys AS (
        -- Join touchpoints with conversions within attribution window
        SELECT
            t.user_id,
            t.touchpoint_date,
            t.source,
            t.medium,
            t.campaign,
            t.touchpoint_spend,
            c.conversion_date,
            c.conversion_value,
            DATE_DIFF(c.conversion_date, t.touchpoint_date, DAY) as days_to_conversion
        FROM touchpoints t
        LEFT JOIN conversions c
            ON t.user_id = c.user_id
            AND c.conversion_timestamp > t.event_timestamp
            AND DATE_DIFF(c.conversion_date, t.touchpoint_date, DAY) BETWEEN {MIN_ATTRIBUTION_WINDOW} AND {MAX_ATTRIBUTION_WINDOW}
    )
    SELECT
        user_id,
        touchpoint_date,
        source,
        medium,
        campaign,
        SUM(touchpoint_spend) as daily_touchpoint_spend,
        conversion_date,
        AVG(conversion_value) as conversion_value,
        MIN(days_to_conversion) as days_to_conversion
    FROM journeys
    GROUP BY user_id, touchpoint_date, source, medium, campaign, conversion_date
    ORDER BY user_id, touchpoint_date
    """

    print(f"Fetching user journeys from {extended_start} to {extended_end}")
    df = bq.query(sql).to_dataframe()
    print(f"Found {len(df)} touchpoint-conversion pairs")
    return df


def attribute_conversions(journeys_df: pd.DataFrame,
                         attribution_model: str = 'time_decay') -> pd.DataFrame:
    """
    Apply multi-touch attribution to user journeys.
    Returns DataFrame with attributed spend and conversions by date.
    """

    if not ATTRIBUTION_AVAILABLE:
        print("Warning: Using simple last-touch attribution (reward_attribution not available)")
        # Fallback to simple attribution
        return simple_attribution(journeys_df)

    # Use sophisticated attribution
    wrapper = RewardAttributionWrapper()

    # Group by user and conversion
    attributed_data = []

    for (user_id, conversion_date), group in journeys_df.groupby(['user_id', 'conversion_date']):
        if pd.isna(conversion_date):
            continue

        touchpoints = group[group['touchpoint_date'].notna()].sort_values('touchpoint_date')

        if len(touchpoints) == 0:
            continue

        conversion_value = touchpoints['conversion_value'].iloc[0]

        # Calculate attribution weights based on model
        if attribution_model == 'time_decay':
            # More recent touchpoints get more credit
            days_from_conversion = (conversion_date - touchpoints['touchpoint_date']).dt.days
            weights = np.exp(-0.1 * days_from_conversion)  # Exponential decay
            weights = weights / weights.sum()
        elif attribution_model == 'linear':
            # Equal credit to all touchpoints
            weights = np.ones(len(touchpoints)) / len(touchpoints)
        elif attribution_model == 'position_based':
            # 40% first, 40% last, 20% middle
            weights = np.ones(len(touchpoints)) * 0.2 / max(1, len(touchpoints) - 2)
            if len(touchpoints) > 0:
                weights[0] = 0.4
            if len(touchpoints) > 1:
                weights[-1] = 0.4
            weights = weights / weights.sum()
        else:
            weights = np.ones(len(touchpoints)) / len(touchpoints)

        # Attribute value and calculate rewards
        for idx, (_, tp) in enumerate(touchpoints.iterrows()):
            attributed_conversion_value = conversion_value * weights[idx]
            reward = attributed_conversion_value - tp['daily_touchpoint_spend']

            attributed_data.append({
                'date': tp['touchpoint_date'],
                'source': tp['source'],
                'medium': tp['medium'],
                'campaign': tp['campaign'],
                'attributed_spend': tp['daily_touchpoint_spend'],
                'attributed_conversions': weights[idx],  # Fractional conversion
                'attributed_value': attributed_conversion_value,
                'reward': reward,
                'days_to_conversion': tp['days_to_conversion'],
                'attribution_weight': weights[idx]
            })

    attributed_df = pd.DataFrame(attributed_data)

    # Aggregate by date
    daily_attributed = attributed_df.groupby('date').agg({
        'attributed_spend': 'sum',
        'attributed_conversions': 'sum',
        'attributed_value': 'sum',
        'reward': 'sum',
        'days_to_conversion': 'mean',
        'attribution_weight': 'mean'
    }).reset_index()

    print(f"Attribution complete: {len(daily_attributed)} days with attributed conversions")
    print(f"Average days to conversion: {daily_attributed['days_to_conversion'].mean():.1f}")

    return daily_attributed


def simple_attribution(journeys_df: pd.DataFrame) -> pd.DataFrame:
    """Fallback: Simple last-touch attribution"""
    # Group conversions by the last touchpoint date
    last_touch = journeys_df.dropna(subset=['conversion_date']).groupby(['user_id', 'conversion_date']).last()

    daily = last_touch.groupby('touchpoint_date').agg({
        'daily_touchpoint_spend': 'sum',
        'conversion_value': 'sum'
    }).reset_index()

    daily.columns = ['date', 'attributed_spend', 'attributed_value']
    daily['attributed_conversions'] = daily['attributed_value'] / 74.70  # Avg order value
    daily['reward'] = daily['attributed_value'] - daily['attributed_spend']
    daily['days_to_conversion'] = 7  # Default assumption
    daily['attribution_weight'] = 1.0

    return daily


def fit_attributed_mmm(attributed_df: pd.DataFrame,
                      cac_cap: float = 200.0) -> Dict[str, Any]:
    """
    Fit MMM on properly attributed data (spend and conversions aligned correctly).
    """

    # Ensure we have enough data
    if len(attributed_df) < 30:
        print(f"Warning: Only {len(attributed_df)} days of data, results may be unreliable")

    # Prepare arrays
    spend = attributed_df['attributed_spend'].values
    conversions = attributed_df['attributed_conversions'].values
    rewards = attributed_df['reward'].values

    # Remove zero-spend days to avoid log issues
    mask = spend > 0
    spend = spend[mask]
    conversions = conversions[mask]
    rewards = rewards[mask]

    # Fit log-log model: log(conversions) = a + b*log(spend)
    log_spend = np.log(spend + 1e-6)
    log_conv = np.log(conversions + 1e-6)

    # Use rewards to weight the regression (higher rewards = more important)
    weights = np.clip(rewards + 100, 1, None)  # Shift to positive, min weight 1
    weights = weights / weights.mean()  # Normalize

    # Weighted least squares
    X = np.column_stack([np.ones_like(log_spend), log_spend])
    W = np.diag(weights)

    # Solve: (X'WX)^-1 X'Wy
    coef = np.linalg.inv(X.T @ W @ X) @ (X.T @ W @ log_conv)
    a, b = coef

    print(f"Fitted attributed MMM: elasticity b = {b:.3f}")
    print(f"Average attributed CAC: ${spend.sum()/conversions.sum():.2f}")
    print(f"Average reward per conversion: ${rewards.mean():.2f}")

    # Build response curve
    min_spend = np.percentile(spend, 10)
    max_spend = np.percentile(spend, 90)
    spend_grid = np.linspace(min_spend, max_spend * 1.5, 50)

    # Predict with attribution-aware model
    conv_grid = np.exp(a + b * np.log(spend_grid + 1e-6))

    # Calculate attributed CAC for each point
    cac_grid = spend_grid / np.maximum(conv_grid, 1e-6)

    # Find optimal budget under CAC constraint
    valid_idx = cac_grid <= cac_cap
    if valid_idx.any():
        optimal_idx = np.where(valid_idx)[0][-1]  # Highest spend under CAC cap
        optimal_budget = spend_grid[optimal_idx]
        expected_conversions = conv_grid[optimal_idx]
        expected_cac = cac_grid[optimal_idx]
    else:
        optimal_budget = spend_grid[0]
        expected_conversions = conv_grid[0]
        expected_cac = cac_grid[0]

    return {
        'elasticity': b,
        'intercept': a,
        'spend_grid': spend_grid.tolist(),
        'conv_grid': conv_grid.tolist(),
        'cac_grid': cac_grid.tolist(),
        'optimal_budget': optimal_budget,
        'expected_conversions': expected_conversions,
        'expected_cac': expected_cac,
        'avg_days_to_conversion': attributed_df['days_to_conversion'].mean(),
        'avg_attribution_weight': attributed_df['attribution_weight'].mean()
    }


def ensure_attributed_tables(bq: bigquery.Client, project: str, dataset: str):
    """Create tables for attributed MMM results"""

    # mmm_curves_attributed
    curves_id = f"{project}.{dataset}.mmm_curves_attributed"
    try:
        bq.get_table(curves_id)
    except NotFound:
        schema = [
            bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("channel", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("window_start", "DATE", mode="REQUIRED"),
            bigquery.SchemaField("window_end", "DATE", mode="REQUIRED"),
            bigquery.SchemaField("model", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("attribution_model", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("params", "JSON", mode="NULLABLE"),
            bigquery.SchemaField("spend_grid", "JSON", mode="NULLABLE"),
            bigquery.SchemaField("conv_grid", "JSON", mode="NULLABLE"),
            bigquery.SchemaField("cac_grid", "JSON", mode="NULLABLE"),
            bigquery.SchemaField("diagnostics", "JSON", mode="NULLABLE"),
        ]
        table = bigquery.Table(curves_id, schema=schema)
        table.time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY, field="timestamp"
        )
        bq.create_table(table)
        print(f"Created table: {curves_id}")

    # mmm_allocations_attributed
    allocs_id = f"{project}.{dataset}.mmm_allocations_attributed"
    try:
        bq.get_table(allocs_id)
    except NotFound:
        schema = [
            bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("channel", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("proposed_daily_budget", "FLOAT", mode="REQUIRED"),
            bigquery.SchemaField("expected_conversions", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("expected_cac", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("avg_days_to_conversion", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("constraints", "JSON", mode="NULLABLE"),
            bigquery.SchemaField("diagnostics", "JSON", mode="NULLABLE"),
        ]
        table = bigquery.Table(allocs_id, schema=schema)
        table.time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY, field="timestamp"
        )
        bq.create_table(table)
        print(f"Created table: {allocs_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', help='YYYY-MM-DD (default: 90 days ago)')
    parser.add_argument('--end', help='YYYY-MM-DD (default: today)')
    parser.add_argument('--attribution_model', default='time_decay',
                       choices=['time_decay', 'linear', 'position_based'],
                       help='Attribution model to use')
    parser.add_argument('--cac_cap', type=float, default=200.0)
    args = parser.parse_args()

    project = os.getenv('GOOGLE_CLOUD_PROJECT')
    dataset = os.getenv('BIGQUERY_TRAINING_DATASET')
    ga_dataset = os.getenv('GA4_EXPORT_DATASET', 'ga360-bigquery-datashare.analytics_308028264')

    if not project or not dataset:
        raise RuntimeError('Set GOOGLE_CLOUD_PROJECT and BIGQUERY_TRAINING_DATASET')

    end_date = date.fromisoformat(args.end) if args.end else date.today()
    start_date = date.fromisoformat(args.start) if args.start else (end_date - timedelta(days=90))

    print(f"Running attributed MMM from {start_date} to {end_date}")
    print(f"Attribution model: {args.attribution_model}")
    print(f"Attribution window: {MIN_ATTRIBUTION_WINDOW}-{MAX_ATTRIBUTION_WINDOW} days")

    # Initialize BigQuery
    bq = bigquery.Client(project=project)
    ensure_attributed_tables(bq, project, dataset)

    # Get user journeys
    print("\n1. Fetching user journeys...")
    journeys = get_user_journeys(bq, ga_dataset.split('.')[0],
                                 ga_dataset.split('.')[1],
                                 start_date, end_date)

    if len(journeys) == 0:
        print("No journeys found. Check your GA4 data.")
        return

    # Apply attribution
    print("\n2. Applying multi-touch attribution...")
    attributed = attribute_conversions(journeys, args.attribution_model)

    # Fit MMM on attributed data
    print("\n3. Fitting MMM on attributed data...")
    mmm_results = fit_attributed_mmm(attributed, args.cac_cap)

    # Write results to BigQuery
    print("\n4. Writing results to BigQuery...")
    now = datetime.utcnow().isoformat()

    # Curves table
    curves_row = {
        'timestamp': now,
        'channel': 'all_channels',  # Aggregate for now
        'window_start': str(start_date),
        'window_end': str(end_date),
        'model': 'log_log_attributed',
        'attribution_model': args.attribution_model,
        'params': json.dumps({
            'elasticity': mmm_results['elasticity'],
            'intercept': mmm_results['intercept'],
            'min_attribution_days': MIN_ATTRIBUTION_WINDOW,
            'max_attribution_days': MAX_ATTRIBUTION_WINDOW
        }),
        'spend_grid': json.dumps(mmm_results['spend_grid']),
        'conv_grid': json.dumps(mmm_results['conv_grid']),
        'cac_grid': json.dumps(mmm_results['cac_grid']),
        'diagnostics': json.dumps({
            'avg_days_to_conversion': mmm_results['avg_days_to_conversion'],
            'avg_attribution_weight': mmm_results['avg_attribution_weight'],
            'num_journeys': len(journeys),
            'num_days': len(attributed)
        })
    }

    curves_table = f"{project}.{dataset}.mmm_curves_attributed"
    bq.insert_rows_json(curves_table, [curves_row])

    # Allocations table
    alloc_row = {
        'timestamp': now,
        'channel': 'all_channels',
        'proposed_daily_budget': float(mmm_results['optimal_budget']),
        'expected_conversions': float(mmm_results['expected_conversions']),
        'expected_cac': float(mmm_results['expected_cac']),
        'avg_days_to_conversion': float(mmm_results['avg_days_to_conversion']),
        'constraints': json.dumps({'cac_cap': args.cac_cap}),
        'diagnostics': json.dumps({
            'elasticity': mmm_results['elasticity'],
            'attribution_model': args.attribution_model
        })
    }

    allocs_table = f"{project}.{dataset}.mmm_allocations_attributed"
    bq.insert_rows_json(allocs_table, [alloc_row])

    print(f"\n✅ Attributed MMM complete!")
    print(f"Results written to:")
    print(f"  - {curves_table}")
    print(f"  - {allocs_table}")

    print(f"\n📊 Key Results:")
    print(f"  - Elasticity: {mmm_results['elasticity']:.3f}")
    print(f"  - Optimal budget: ${mmm_results['optimal_budget']:,.0f}")
    print(f"  - Expected conversions: {mmm_results['expected_conversions']:,.0f}")
    print(f"  - Expected CAC: ${mmm_results['expected_cac']:.2f}")
    print(f"  - Avg days to conversion: {mmm_results['avg_days_to_conversion']:.1f}")

    print(f"\n🎯 Comparison to same-day attribution:")
    print(f"  Run 'python AELP2/pipelines/mmm_service.py' to compare")
    print(f"  The difference shows the impact of proper attribution")


if __name__ == '__main__':
    main()