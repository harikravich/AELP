#!/usr/bin/env python3
"""
MCC Coordinator: Orchestrate Ads data loads across all child accounts.

Runs selected Ads loaders for each child account under the MCC, with simple
rate limiting between accounts to respect API quotas.

Requirements:
- Env: GOOGLE_CLOUD_PROJECT, BIGQUERY_TRAINING_DATASET
- Ads env: GOOGLE_ADS_DEVELOPER_TOKEN, GOOGLE_ADS_CLIENT_ID, GOOGLE_ADS_CLIENT_SECRET,
  GOOGLE_ADS_REFRESH_TOKEN, GOOGLE_ADS_LOGIN_CUSTOMER_ID (MCC)

Usage:
  python -m AELP2.pipelines.ads_mcc_coordinator --start 2024-07-01 --end 2024-07-31 \
    --tasks campaigns,keywords,search_terms,geo_device,adgroups,conversion_actions

Env controls:
- AELP2_ADS_MCC_DELAY_SECONDS: delay between accounts (default: 2.0 seconds)
"""

import os
import time
import argparse
import logging
from datetime import datetime
from typing import List, Callable

from AELP2.pipelines.google_ads_mcc_to_bq import enumerate_child_accounts
from AELP2.pipelines.google_ads_to_bq import run as load_campaigns
from AELP2.pipelines.google_ads_keywords_to_bq import run as load_keywords
from AELP2.pipelines.google_ads_search_terms_to_bq import run as load_search_terms
from AELP2.pipelines.google_ads_geo_device_to_bq import run as load_geo_device
from AELP2.pipelines.google_ads_adgroups_to_bq import run as load_adgroups
from AELP2.pipelines.google_ads_ad_performance_to_bq import run as load_ad_performance
from AELP2.pipelines.google_ads_conversion_actions_to_bq import run as load_conversion_actions
from AELP2.pipelines.google_ads_conversion_stats_by_action_to_bq import run as load_conversion_action_stats

logger = logging.getLogger(__name__)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))


def _validate_dates(start: str, end: str):
    datetime.strptime(start, "%Y-%m-%d")
    datetime.strptime(end, "%Y-%m-%d")


def _split_tasks(csv: str) -> List[str]:
    tasks = [t.strip() for t in csv.split(',') if t.strip()]
    valid = {"campaigns", "keywords", "search_terms", "geo_device", "adgroups", "conversion_actions", "conversion_action_stats", "ad_performance"}
    for t in tasks:
        if t not in valid:
            raise ValueError(f"Invalid task '{t}'. Valid: {sorted(valid)}")
    return tasks


def _should_retry_error(e: Exception) -> bool:
    msg = str(e)
    # Retry on transient Ads API internal errors
    return 'Internal error encountered' in msg or 'Retry the request' in msg


def _run_with_retry(fn: Callable, *args, **kwargs):
    retries = int(os.getenv('AELP2_ADS_API_RETRIES', '1'))
    delay = float(os.getenv('AELP2_ADS_API_RETRY_DELAY', '2.0'))
    attempt = 0
    while True:
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if attempt >= retries or not _should_retry_error(e):
                raise
            attempt += 1
            logging.warning(f"Transient Ads API error, retrying {attempt}/{retries} in {delay}s: {e}")
            time.sleep(delay)


def run(start: str, end: str, tasks: List[str], *, only: List[str] | None = None, skip: List[str] | None = None):
    _validate_dates(start, end)
    delay_s = float(os.getenv('AELP2_ADS_MCC_DELAY_SECONDS', '2.0'))

    cids = enumerate_child_accounts()
    if only:
        only_set = {cid.replace('-', '') for cid in only}
        cids = [c for c in cids if c in only_set]
    if skip:
        skip_set = {cid.replace('-', '') for cid in skip}
        cids = [c for c in cids if c not in skip_set]
    if not cids:
        logger.warning("No child accounts found under MCC.")
        return 0
    logger.info(f"Loading {tasks} for {len(cids)} accounts, window {start}..{end} (delay {delay_s}s)")

    for idx, cid in enumerate(cids, 1):
        logger.info(f"[{idx}/{len(cids)}] Customer {cid}")
        try:
            if 'campaigns' in tasks:
                _run_with_retry(load_campaigns, start, end, customer_id=cid)
            if 'keywords' in tasks:
                _run_with_retry(load_keywords, start, end, customer_id=cid)
            if 'search_terms' in tasks:
                _run_with_retry(load_search_terms, start, end, customer_id=cid)
            if 'geo_device' in tasks:
                _run_with_retry(load_geo_device, start, end, customer_id=cid)
            if 'adgroups' in tasks:
                _run_with_retry(load_adgroups, start, end, customer_id=cid)
            if 'conversion_actions' in tasks:
                _run_with_retry(load_conversion_actions, customer_id=cid)
            if 'conversion_action_stats' in tasks:
                _run_with_retry(load_conversion_action_stats, start, end, customer_id=cid)
            if 'ad_performance' in tasks:
                _run_with_retry(load_ad_performance, start, end, customer_id=cid)
        except Exception as e:
            logger.error(f"Failed loading for customer {cid}: {e}")
        time.sleep(delay_s)
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--tasks", default="campaigns,keywords,search_terms,geo_device,adgroups,conversion_actions")
    p.add_argument("--only", help="Comma-separated list of customer IDs to include", default="")
    p.add_argument("--skip", help="Comma-separated list of customer IDs to skip", default="")
    args = p.parse_args()
    only = [c.strip() for c in args.only.split(',') if c.strip()]
    skip = [c.strip() for c in args.skip.split(',') if c.strip()]
    exit(run(args.start, args.end, _split_tasks(args.tasks), only=only or None, skip=skip or None))
