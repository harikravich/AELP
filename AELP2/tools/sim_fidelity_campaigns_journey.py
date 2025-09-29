#!/usr/bin/env python3
"""
Phase 3 fidelity: journey stages + recency proxy

Extends Phase 2 with a simple stage model:
- Stage shares inferred from frequency: higher freq -> more 'decision' stage
- Stage multipliers: awareness 0.35x, consideration 0.7x, decision 1.0x (clamped)
- Combine with weekday and frequency fatigue effects

Outputs
- JSON: AELP2/reports/sim_fidelity_campaigns_journey.json
"""
from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from datetime import date, timedelta, datetime
from typing import Dict, List, Tuple

import numpy as np
import requests

META_BASE = "https://graph.facebook.com/v21.0"


def _read_env():
    tok = os.getenv("META_ACCESS_TOKEN")
    acct = os.getenv("META_ACCOUNT_ID")
    if not (tok and acct) and os.path.exists(".env"):
        with open(".env", "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if ln.startswith("export META_ACCESS_TOKEN=") and not tok:
                    tok = ln.split("=", 1)[1].strip()
                if ln.startswith("export META_ACCOUNT_ID=") and not acct:
                    acct = ln.split("=", 1)[1].strip()
    if not tok or not acct:
        raise RuntimeError("Missing META_ACCESS_TOKEN or META_ACCOUNT_ID")
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
    out: List[dict] = []
    pages = 0
    while True:
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        js = r.json()
        out.extend(js.get("data", []))
        nxt = (js.get("paging") or {}).get("next")
        if not nxt:
            break
        url = nxt
        params = None
        pages += 1
        if pages > 1000:
            break
    return out


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


def _split_train_test(dates: List[str]):
    ds = sorted(set(dates))
    train = set(ds[:14])
    test = set(ds[14:21])
    hold = set(ds[21:])
    return train, test, hold


def _fit_lognorm(cpcs: List[float]):
    x = np.array([c for c in cpcs if c > 0], dtype=float)
    if x.size == 0:
        return math.log(2.0), 0.5, 0.5, 8.0
    p5 = float(np.percentile(x, 5))
    p95 = float(np.percentile(x, 95))
    x = np.clip(x, max(1e-6, p5), max(p95, p5 + 1e-3))
    logs = np.log(x)
    mu = float(np.mean(logs))
    sigma = float(np.std(logs) if logs.size > 1 else 0.3)
    sigma = max(0.05, min(sigma, 1.5))
    return mu, sigma, p5, p95


def _beta_hyper_from_global(success: float, total: float, k: float = 150.0):
    rate = (success / total) if total > 0 else 0.01
    rate = min(max(rate, 1e-5), 0.5)
    return rate * k, (1 - rate) * k


def _weekday(dt_str: str) -> int:
    return datetime.strptime(dt_str, "%Y-%m-%d").weekday()


def _journey_multiplier_from_freq(freq: float) -> float:
    # Stage shares from frequency: as freq rises, we weight more to decision
    f = max(0.5, min(5.0, float(freq)))
    # heuristic shares; awareness dominates near 1.0, decision grows with f
    s_decision = np.clip((f - 1.0) / 3.0, 0.1, 0.6)
    s_awareness = np.clip(1.0 - s_decision - 0.3, 0.1, 0.7)
    s_consider = max(0.0, 1.0 - s_awareness - s_decision)
    m = 0.35 * s_awareness + 0.70 * s_consider + 1.00 * s_decision
    return float(np.clip(m, 0.4, 1.0))


def _simulate_day(spend: float, mu: float, sigma: float, cpc_lo: float, cpc_hi: float,
                  alpha: float, beta: float, wd_mult: float, freq_eff: float, journey_mult: float,
                  draws: int = 200):
    spend = max(0.0, float(spend))
    if spend <= 0:
        return 0.0, 0.0, 0.0
    cpcs = np.random.lognormal(mu, sigma, size=draws)
    cpcs = np.clip(cpcs, max(1e-3, cpc_lo), max(cpc_hi, cpc_lo + 1e-3))
    lam_clicks = spend / cpcs
    max_clicks_cap = np.minimum(50000, (spend / np.maximum(0.1, cpc_lo)) + 1000)
    clicks = np.random.poisson(lam=np.minimum(lam_clicks, max_clicks_cap))
    p_cvr = np.random.beta(alpha, beta, size=draws) * wd_mult * freq_eff * journey_mult
    p_cvr = np.clip(p_cvr, 1e-5, 0.5)
    purch = np.array([np.random.binomial(int(cl), p) if cl > 0 else 0 for cl, p in zip(clicks, p_cvr)], dtype=float)
    return float(np.median(purch)), float(np.percentile(purch, 10)), float(np.percentile(purch, 90))


def main():
    token, acct = _read_env()
    today = date.today()
    until = today - timedelta(days=1)
    since = until - timedelta(days=27)
    rows = _get_insights_daily(token, acct, since.isoformat(), until.isoformat())

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
    for v in by_camp.values():
        v.sort(key=lambda x: x["date"])

    train_days, test_days, hold = _split_train_test(all_dates)

    # Global priors and CPC
    global_clicks = 0.0
    global_purch = 0.0
    all_cpcs_train: List[float] = []
    for rows in by_camp.values():
        for r in rows:
            if r["date"] in train_days:
                global_clicks += r["clicks"]
                global_purch += r["purch"]
                if r["clicks"] > 0 and r["spend"] > 0:
                    all_cpcs_train.append(r["spend"] / r["clicks"])
    a0, b0 = _beta_hyper_from_global(global_purch, global_clicks, k=150.0)
    g_mu, g_sigma, g_p5, g_p95 = _fit_lognorm(all_cpcs_train or [2.0, 2.2, 2.5])

    # Weekday mults, freq slope, med freq per campaign
    acc_rate = (global_purch / global_clicks) if global_clicks > 0 else 0.01
    acc_wd_counts = {w: {"succ": 0.0, "tot": 0.0} for w in range(7)}
    for rows in by_camp.values():
        for r in rows:
            if r["date"] not in train_days:
                continue
            w = datetime.strptime(r["date"], "%Y-%m-%d").weekday()
            acc_wd_counts[w]["succ"] += r["purch"]
            acc_wd_counts[w]["tot"] += max(r["clicks"], 0)
    acc_wd_rate = {}
    for w in range(7):
        tot = acc_wd_counts[w]["tot"]
        succ = acc_wd_counts[w]["succ"]
        r = (succ / tot) if tot > 0 else acc_rate
        acc_wd_rate[w] = r / max(acc_rate, 1e-6)

    params = {}
    wd_mults = {}
    freq_slope = {}
    med_freq = {}

    for cid, rows in by_camp.items():
        cpcs = [r["spend"] / r["clicks"] for r in rows if r["date"] in train_days and r["clicks"] > 0 and r["spend"] > 0]
        if len(cpcs) >= 3:
            mu, sigma, p5, p95 = _fit_lognorm(cpcs)
        else:
            mu, sigma, p5, p95 = g_mu, g_sigma, g_p5, g_p95
        clicks_tot = sum(r["clicks"] for r in rows if r["date"] in train_days)
        purch_tot = sum(r["purch"] for r in rows if r["date"] in train_days)
        alpha = a0 + purch_tot
        beta = b0 + max(0.0, clicks_tot - purch_tot)
        # weekday mults
        wd_counts = {w: {"succ": 0.0, "tot": 0.0} for w in range(7)}
        for r in rows:
            if r["date"] in train_days:
                w = datetime.strptime(r["date"], "%Y-%m-%d").weekday()
                wd_counts[w]["succ"] += r["purch"]
                wd_counts[w]["tot"] += max(r["clicks"], 0)
        base_rate = (purch_tot / clicks_tot) if clicks_tot > 0 else acc_rate
        wmult = {}
        for w in range(7):
            tot = wd_counts[w]["tot"]
            succ = wd_counts[w]["succ"]
            local = (succ / tot) if tot > 0 else base_rate
            ratio_local = local / max(base_rate, 1e-6)
            weight = min(1.0, tot / 500.0)
            ratio = weight * ratio_local + (1 - weight) * acc_wd_rate[w]
            wmult[w] = float(np.clip(ratio, 0.7, 1.3))
        wd_mults[cid] = wmult
        # frequency slope
        freqs = [r["freq"] for r in rows if r["date"] in train_days and r["clicks"] > 0]
        m_f = float(np.median(freqs)) if freqs else 1.5
        xs, ys = [], []
        cvrs_pos = [ (rr["purch"]/rr["clicks"]) for rr in rows if rr["date"] in train_days and rr["clicks"]>0 and (rr["purch"]/rr["clicks"])>0 ]
        eps = float(np.percentile(np.array(cvrs_pos), 5)) if len(cvrs_pos) >= 1 else 1e-5
        eps = max(1e-6, eps)
        for r in rows:
            if r["date"] not in train_days or r["clicks"] <= 0:
                continue
            cvr = r["purch"] / r["clicks"]
            xs.append(r["freq"] - m_f)
            ys.append(math.log(max(cvr, 0) + eps))
        if len(xs) >= 5:
            X = np.vstack([np.ones(len(xs)), np.array(xs)]).T
            coef, *_ = np.linalg.lstsq(X, np.array(ys), rcond=None)
            slope = float(coef[1])
        else:
            slope = 0.0
        slope = float(np.clip(slope, -0.25, 0.05))
        freq_slope[cid] = slope
        med_freq[cid] = m_f
        params[cid] = {"mu": mu, "sigma": sigma, "cpc_p5": p5, "cpc_p95": p95, "alpha": alpha, "beta": beta, "name": rows[0]["name"] if rows else cid}

    # simulate
    train_sorted = sorted(train_days)
    days_sorted = sorted(test_days)
    day_actual = {d: {"spend": 0.0, "purch": 0.0} for d in days_sorted}
    day_pred = {d: {"med": 0.0, "p10": 0.0, "p90": 0.0} for d in days_sorted}
    for cid, rows in by_camp.items():
        p = params[cid]
        for r in rows:
            d = r["date"]
            if d not in test_days:
                continue
            spend = r["spend"]
            day_actual[d]["spend"] += spend
            day_actual[d]["purch"] += r["purch"]
            w = datetime.strptime(d, "%Y-%m-%d").weekday()
            wd_mult = wd_mults[cid][w]
            f = r["freq"]
            slope = freq_slope[cid]
            freq_eff = math.exp(slope * (f - med_freq[cid]))
            freq_eff = float(np.clip(freq_eff, 0.7, 1.2))
            journey_mult = _journey_multiplier_from_freq(f)
            med, p10, p90 = _simulate_day(spend, p["mu"], p["sigma"], p["cpc_p5"], p["cpc_p95"], p["alpha"], p["beta"], wd_mult, freq_eff, journey_mult, draws=200)
            day_pred[d]["med"] += med
            day_pred[d]["p10"] += p10
            day_pred[d]["p90"] += p90

    purch_mape, cac_mape, cov_hits, cov_tot = [], [], 0, 0
    daily_rows = []
    for d in days_sorted:
        a = day_actual[d]
        p = day_pred[d]
        spend = float(a["spend"]) 
        actual_p = float(a["purch"]) 
        pred_p = max(0.0, float(p["med"]))
        if actual_p > 0:
            purch_mape.append(abs(pred_p - actual_p) / actual_p)
        actual_cac = spend / actual_p if actual_p > 0 else float("inf")
        pred_cac = spend / pred_p if pred_p > 0 else float("inf")
        if math.isfinite(actual_cac) and math.isfinite(pred_cac) and actual_cac > 0:
            cac_mape.append(abs(pred_cac - actual_cac) / actual_cac)
        lo, hi = float(p["p10"]), float(p["p90"])
        if actual_p >= lo and actual_p <= hi:
            cov_hits += 1
        cov_tot += 1
        daily_rows.append({"date": d, "spend": round(spend, 2), "actual_purch": round(actual_p, 2), "pred_purch_med": round(pred_p, 2), "pred_purch_p10": round(lo, 2), "pred_purch_p90": round(hi, 2), "actual_cac": (round(actual_cac, 2) if math.isfinite(actual_cac) else None), "pred_cac": (round(pred_cac, 2) if math.isfinite(pred_cac) else None)})

    out = {
        "summary": {
            "purchases_day_mape": (round(float(np.mean(purch_mape) * 100), 2) if purch_mape else None),
            "cac_day_mape": (round(float(np.mean(cac_mape) * 100), 2) if cac_mape else None),
            "coverage80": (round(100.0 * cov_hits / cov_tot, 1) if cov_tot else None),
        },
        "daily": daily_rows,
    }

    os.makedirs("AELP2/reports", exist_ok=True)
    path = "AELP2/reports/sim_fidelity_campaigns_journey.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(path)


if __name__ == "__main__":
    np.random.seed(int(os.getenv("AELP2_SEED", "44")))
    main()

