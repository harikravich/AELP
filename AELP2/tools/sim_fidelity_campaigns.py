#!/usr/bin/env python3
"""
Phase 1 fidelity: per-campaign heterogeneity

What it does
- Pulls Meta daily insights (by campaign, last 28 days, daily rows)
- Splits into train (first 14 days) and test (next 7 days)
- Fits per-campaign CPC distributions (lognormal, clamped to p5–p95)
- Fits per-campaign CVR priors (Beta with partial pooling to account-level)
- Simulates purchases for test days given actual spend, aggregates to account/day
- Emits accuracy (MAPE) for purchases/day and CAC/day + 80% interval coverage

Outputs
- JSON: AELP2/reports/sim_fidelity_campaigns.json

Safety / non-hang
- Hard caps on pages, requests, and simulation draws; timeouts on HTTP
- Clamp CPC to reasonable percentiles; cap max clicks per day
"""
from __future__ import annotations

import json
import math
import os
import statistics
from collections import defaultdict
from datetime import date, timedelta
from typing import Dict, List, Tuple

import numpy as np
import requests


META_BASE = "https://graph.facebook.com/v21.0"


def _read_env() -> Tuple[str, str]:
    # Prefer process env; fallback to .env file in CWD
    tok = os.getenv("META_ACCESS_TOKEN")
    acct = os.getenv("META_ACCOUNT_ID")
    if not (tok and acct) and os.path.exists(".env"):
        # simple parse: lines like export VAR=value
        with open(".env", "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if ln.startswith("export META_ACCESS_TOKEN=") and not tok:
                    tok = ln.split("=", 1)[1].strip()
                if ln.startswith("export META_ACCOUNT_ID=") and not acct:
                    acct = ln.split("=", 1)[1].strip()
    if not tok or not acct:
        raise RuntimeError("Missing META_ACCESS_TOKEN or META_ACCOUNT_ID in env/.env")
    return tok, acct


def _get_insights_daily(token: str, acct: str, since: str, until: str) -> List[dict]:
    url = f"{META_BASE}/{acct}/insights"
    params = {
        "time_increment": 1,
        "level": "campaign",
        "fields": (
            "campaign_id,campaign_name,date_start,impressions,clicks,spend,frequency,actions"
        ),
        "time_range": json.dumps({"since": since, "until": until}),
        "access_token": token,
    }
    rows: List[dict] = []
    pages = 0
    while True:
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        js = r.json()
        rows.extend(js.get("data", []))
        nxt = (js.get("paging") or {}).get("next")
        if not nxt:
            break
        url = nxt
        params = None
        pages += 1
        if pages > 1000:
            break
    return rows


def _extract_purchase_count(actions: List[dict] | None) -> float:
    if not actions:
        return 0.0
    for a in actions:
        if a.get("action_type") == "offsite_conversion.fb_pixel_purchase":
            try:
                return float(a.get("value") or 0)
            except Exception:
                return 0.0
    return 0.0


def _split_train_test(dates: List[str]) -> Tuple[set, set, set]:
    # dates are ISO strings; sort ascending
    ds = sorted(set(dates))
    if len(ds) < 21:
        raise RuntimeError("Need at least 21 daily rows to split 14→7→(7 holdout)")
    train = set(ds[:14])
    test = set(ds[14:21])
    hold = set(ds[21:])
    return train, test, hold


def _fit_lognorm(cpcs: List[float]) -> Tuple[float, float, float, float]:
    # Return mu, sigma, p5, p95
    x = np.array([c for c in cpcs if c > 0], dtype=float)
    if x.size == 0:
        # fallback default
        return math.log(2.0), 0.5, 0.5, 8.0
    p5 = float(np.percentile(x, 5))
    p95 = float(np.percentile(x, 95))
    x = np.clip(x, max(1e-6, p5), max(p95, p5 + 1e-3))
    logs = np.log(x)
    mu = float(np.mean(logs))
    sigma = float(np.std(logs) if logs.size > 1 else 0.3)
    sigma = max(0.05, min(sigma, 1.5))
    return mu, sigma, p5, p95


def _beta_hyper_from_global(success: float, total: float, k: float = 100.0) -> Tuple[float, float]:
    # Convert global rate into alpha0,beta0 with prior strength k
    rate = (success / total) if total > 0 else 0.01
    rate = min(max(rate, 1e-5), 0.5)
    alpha0 = rate * k
    beta0 = (1 - rate) * k
    return alpha0, beta0


def _simulate_day(spend: float, mu: float, sigma: float, cpc_lo: float, cpc_hi: float,
                  alpha: float, beta: float, draws: int = 200) -> Tuple[float, float, float]:
    # Returns (median_purch, p10_purch, p90_purch)
    spend = max(0.0, float(spend))
    if spend <= 0.0:
        return 0.0, 0.0, 0.0
    # Sample CPCs
    cpcs = np.random.lognormal(mean=mu, sigma=sigma, size=draws)
    cpcs = np.clip(cpcs, max(1e-3, cpc_lo), max(cpc_hi, cpc_lo + 1e-3))
    # Expected clicks and sampled
    lam_clicks = spend / cpcs
    max_clicks_cap = np.minimum(50000, (spend / np.maximum(0.1, cpc_lo)) + 1000)
    clicks = np.random.poisson(lam=np.minimum(lam_clicks, max_clicks_cap))
    # Sample CVR from Beta, bound it
    p_cvr = np.random.beta(alpha, beta, size=draws)
    p_cvr = np.clip(p_cvr, 1e-5, 0.5)
    purch = np.array([np.random.binomial(int(cl), p) if cl > 0 else 0 for cl, p in zip(clicks, p_cvr)], dtype=float)
    return float(np.median(purch)), float(np.percentile(purch, 10)), float(np.percentile(purch, 90))


def main():
    token, acct = _read_env()
    today = date.today()
    # Use last 28 days ending yesterday
    until = today - timedelta(days=1)
    since = until - timedelta(days=27)
    rows = _get_insights_daily(token, acct, since.isoformat(), until.isoformat())

    # Organize by campaign/date
    by_camp: Dict[str, List[dict]] = defaultdict(list)
    all_dates: List[str] = []
    for r in rows:
        cid = r.get("campaign_id")
        if not cid:
            continue
        d = r.get("date_start")
        all_dates.append(d)
        by_camp[cid].append({
            "date": d,
            "name": r.get("campaign_name"),
            "spend": float(r.get("spend") or 0.0),
            "impr": int(float(r.get("impressions") or 0)),
            "clicks": int(float(r.get("clicks") or 0)),
            "freq": float(r.get("frequency") or 0.0),
            "purch": _extract_purchase_count(r.get("actions")),
        })
    # Sort per campaign by date
    for v in by_camp.values():
        v.sort(key=lambda x: x["date"])

    train_days, test_days, hold_days = _split_train_test(all_dates)

    # Fit global and per-campaign params (train window)
    global_clicks = 0.0
    global_purch = 0.0
    all_cpcs_train: List[float] = []
    for cid, rows in by_camp.items():
        for r in rows:
            if r["date"] not in train_days:
                continue
            clicks = r["clicks"]
            purch = r["purch"]
            spend = r["spend"]
            if clicks > 0 and spend > 0:
                all_cpcs_train.append(spend / clicks)
            global_clicks += clicks
            global_purch += purch

    alpha0, beta0 = _beta_hyper_from_global(global_purch, global_clicks, k=150.0)

    per_camp_params = {}
    # Account-level CPC fallback
    g_mu, g_sigma, g_p5, g_p95 = _fit_lognorm(all_cpcs_train or [2.0, 2.2, 2.5])

    for cid, rows in by_camp.items():
        # Train-only data for this campaign
        cpcs = []
        clicks_tot = 0.0
        purch_tot = 0.0
        for r in rows:
            if r["date"] in train_days:
                if r["clicks"] > 0 and r["spend"] > 0:
                    cpcs.append(r["spend"] / r["clicks"])
                clicks_tot += r["clicks"]
                purch_tot += r["purch"]
        if len(cpcs) >= 3:
            mu, sigma, p5, p95 = _fit_lognorm(cpcs)
        else:
            mu, sigma, p5, p95 = g_mu, g_sigma, g_p5, g_p95
        alpha = alpha0 + purch_tot
        beta = beta0 + max(0.0, clicks_tot - purch_tot)
        per_camp_params[cid] = {
            "mu": mu,
            "sigma": sigma,
            "cpc_p5": p5,
            "cpc_p95": p95,
            "alpha": alpha,
            "beta": beta,
            "name": rows[0]["name"] if rows else cid,
        }

    # Simulate test window day-by-day
    days_sorted = sorted(test_days)
    day_actual = {d: {"spend": 0.0, "purch": 0.0} for d in days_sorted}
    day_pred = {d: {"med": 0.0, "p10": 0.0, "p90": 0.0} for d in days_sorted}

    for cid, rows in by_camp.items():
        p = per_camp_params[cid]
        for r in rows:
            d = r["date"]
            if d not in test_days:
                continue
            spend = r["spend"]
            day_actual[d]["spend"] += spend
            day_actual[d]["purch"] += r["purch"]
            med, p10, p90 = _simulate_day(spend, p["mu"], p["sigma"], p["cpc_p5"], p["cpc_p95"], p["alpha"], p["beta"], draws=200)
            day_pred[d]["med"] += med
            day_pred[d]["p10"] += p10
            day_pred[d]["p90"] += p90

    # Metrics
    purch_mape_list = []
    cac_mape_list = []
    coverage_hits = 0
    coverage_total = 0

    daily_rows = []
    for d in days_sorted:
        a = day_actual[d]
        p = day_pred[d]
        actual_p = float(a["purch"])  # counts
        spend = float(a["spend"])     # $
        pred_p = max(0.0, float(p["med"]))
        # MAPE purchases/day
        if actual_p > 0:
            purch_mape_list.append(abs(pred_p - actual_p) / actual_p)
        # CAC/day
        actual_cac = spend / actual_p if actual_p > 0 else float("inf")
        pred_cac = spend / pred_p if pred_p > 0 else float("inf")
        if math.isfinite(actual_cac) and math.isfinite(pred_cac) and actual_cac > 0:
            cac_mape_list.append(abs(pred_cac - actual_cac) / actual_cac)
        # coverage (80%)
        lo, hi = float(p["p10"]), float(p["p90"])
        if actual_p >= lo and actual_p <= hi:
            coverage_hits += 1
        coverage_total += 1
        daily_rows.append({
            "date": d,
            "spend": round(spend, 2),
            "actual_purch": round(actual_p, 2),
            "pred_purch_med": round(pred_p, 2),
            "pred_purch_p10": round(lo, 2),
            "pred_purch_p90": round(hi, 2),
            "actual_cac": (round(actual_cac, 2) if math.isfinite(actual_cac) else None),
            "pred_cac": (round(pred_cac, 2) if math.isfinite(pred_cac) else None),
        })

    out = {
        "window": {
            "since": since.isoformat(),
            "until": until.isoformat(),
            "train_days": sorted(train_days),
            "test_days": days_sorted,
        },
        "summary": {
            "purchases_day_mape": (round(float(np.mean(purch_mape_list) * 100), 2) if purch_mape_list else None),
            "cac_day_mape": (round(float(np.mean(cac_mape_list) * 100), 2) if cac_mape_list else None),
            "coverage80": (round(100.0 * coverage_hits / coverage_total, 1) if coverage_total else None),
        },
        "daily": daily_rows,
        "per_campaign": per_camp_params,
    }

    os.makedirs("AELP2/reports", exist_ok=True)
    path = "AELP2/reports/sim_fidelity_campaigns.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(path)


if __name__ == "__main__":
    np.random.seed(int(os.getenv("AELP2_SEED", "42")))
    main()

