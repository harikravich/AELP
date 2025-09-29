#!/usr/bin/env python3
"""
Impact.com daily performance backfill (auto-discovery).

Steps:
 1) List advertiser reports and pick candidates that look daily (contain "Performance" and "Date").
 2) For each candidate, try fetching CSV month-by-month over the requested window with several date param styles.
 3) On first candidate that returns non-empty rows, backfill months into
    <project>.<dataset>.impact_partner_performance (partitioned by date), upserting per month.

Env:
  IMPACT_ACCOUNT_SID, IMPACT_AUTH_TOKEN (or IMPACT_BEARER_TOKEN)
  GOOGLE_CLOUD_PROJECT, BIGQUERY_TRAINING_DATASET
  Optional: IMPACT_SUBAID

Usage:
  python3 -m AELP2.pipelines.impact_backfill_performance --months 12
"""
from __future__ import annotations
import os, sys, time, csv, io
import argparse
from datetime import date, timedelta
from typing import List, Dict, Any, Tuple

from google.cloud import bigquery

from AELP2.pipelines.impact_to_bq import list_reports, fetch_report_csv, normalize_rows, ensure_table, delete_range


def month_bounds(d: date) -> Tuple[str, str]:
    start = d.replace(day=1)
    # next month
    if start.month == 12:
        next_start = start.replace(year=start.year+1, month=1, day=1)
    else:
        next_start = start.replace(month=start.month+1, day=1)
    end = next_start - timedelta(days=1)
    return start.isoformat(), end.isoformat()


def candidate_reports() -> List[Tuple[str, str]]:
    reps = list_reports()
    cands: List[Tuple[str,str]] = []
    for r in reps:
        rid = r.get('Id') or r.get('id') or r.get('name')
        nm = r.get('Name') or r.get('name') or ''
        low = nm.lower()
        if 'performance' in low and (('date' in low) or ('day' in low)) and 'month' not in low and 'week' not in low:
            cands.append((str(rid), nm))
    # Prefer fine-grained report if present
    prefs = [
        'adv_performance_by_ad_media_date',
        'att_adv_performance_by_day_pm_only',
        'att_adv_performance_by_media_pm_only',
        'adv_performance_by_partner_date',
    ]
    cands.sort(key=lambda x: prefs.index(x[0]) if x[0] in prefs else 999)
    return cands


def try_fetch(report_id: str, start: str, end: str) -> List[Dict[str,Any]]:
    try:
        csv_text = fetch_report_csv(report_id, start, end)
        rows = normalize_rows(csv_text)
        return rows
    except SystemExit as e:
        # report not permitted etc.
        return []


def backfill(report_id: str, months: int, project: str, dataset: str) -> int:
    bq = bigquery.Client(project=project)
    table_id = ensure_table(bq, project, dataset)
    today = date.today().replace(day=1)  # start from current month
    total = 0
    for i in range(months):
        m = today - timedelta(days=30*i)
        start, end = month_bounds(m)
        # Avoid partial month: clamp end to yesterday
        y = date.today() - timedelta(days=1)
        if end > y:
            end = y.isoformat()
        rows = try_fetch(report_id, start, end)
        if not rows:
            continue
        delete_range(bq, table_id, start, end)
        # Coerce date to YYYY-MM-DD for BQ DATE
        for r in rows:
            s = str(r.get('date'))
            if '/' in s:  # MM/DD/YYYY → YYYY-MM-DD
                mm, dd, yy = s.split('/')
                r['date'] = f"{yy}-{int(mm):02d}-{int(dd):02d}"
        errors = bq.insert_rows_json(table_id, rows)
        if errors:
            raise SystemExit(f"BQ insert errors: {errors}")
        total += len(rows)
        time.sleep(0.3)
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--months', type=int, default=12)
    args = ap.parse_args()

    project = os.getenv('GOOGLE_CLOUD_PROJECT')
    dataset = os.getenv('BIGQUERY_TRAINING_DATASET')
    if not project or not dataset:
        raise SystemExit('Set GOOGLE_CLOUD_PROJECT and BIGQUERY_TRAINING_DATASET')

    cands = candidate_reports()
    if not cands:
        print('No API-enabled daily performance reports visible.')
        sys.exit(0)
    print('[impact] Candidate daily reports:')
    for rid, name in cands[:15]:
        print(' -', rid, '|', name)

    # Probe each candidate on last full month
    last_full = (date.today().replace(day=1) - timedelta(days=1)).replace(day=1)
    s, e = month_bounds(last_full)
    chosen = None
    rows_seen = 0
    for rid, _ in cands:
        rows = try_fetch(rid, s, e)
        if rows:
            chosen = rid
            rows_seen = len(rows)
            break
    if not chosen:
        print('[impact] No non-empty candidate found (check API Accessible toggle or filters).')
        sys.exit(0)
    print(f"[impact] Using report {chosen} (sample rows={rows_seen})")

    total = backfill(chosen, args.months, project, dataset)
    print(f"[impact] Backfill complete: {total} rows written to {project}.{dataset}.impact_partner_performance")


if __name__ == '__main__':
    main()
