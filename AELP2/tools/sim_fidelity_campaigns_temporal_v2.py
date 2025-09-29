#!/usr/bin/env python3
"""
Temporal simulator v2: time-decayed CVR priors + auto window (per-campaign)

Features in this step:
- Auto-select train window per campaign (7/14/21) based on drift & support
- Time-decayed counts in Beta CVR prior (half-life configurable)

Outputs
- JSON: AELP2/reports/sim_fidelity_campaigns_temporal_v2.json
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
    tok = os.getenv("META_ACCESS_TOKEN") or os.getenv("META_ACCESS_TOKEN_DISABLED")
    acct = os.getenv("META_ACCOUNT_ID")
    if not (tok and acct) and os.path.exists(".env"):
        with open(".env", "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if ln.startswith("export META_ACCESS_TOKEN=") and not tok:
                    tok = ln.split("=", 1)[1].strip()
                if ln.startswith("export META_ACCESS_TOKEN_DISABLED=") and not tok:
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


def _get_campaign_creatives_created(token: str, campaign_id: str, max_pages: int = 5) -> List[str]:
    """Return list of ISO created_time strings for ads under a campaign."""
    url = f"{META_BASE}/{campaign_id}/ads"
    params = {
        "fields": "id,created_time,updated_time,effective_status",
        "limit": 200,
        "access_token": token,
    }
    times: List[str] = []
    pages = 0
    while True:
        r = requests.get(url, params=params if pages == 0 else None, timeout=60)
        if r.status_code >= 400:
            break
        js = r.json()
        for row in js.get("data", []):
            ct = row.get("created_time") or row.get("updated_time")
            if ct:
                times.append(ct[:10])
        nxt = (js.get("paging") or {}).get("next")
        if not nxt:
            break
        url = nxt
        pages += 1
        if pages >= max_pages:
            break
    return times


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


def _auto_window(dates: List[str], clicks: List[float], purch: List[float]) -> Tuple[set, set]:
    # dates sorted ascending; choose 7/14/21-day train window ending at index len-8
    ds = dates
    n = len(ds)
    # ensure we have at least 28 days total
    # train end index = n-8 (leave 7 days test at end)
    end_idx = n - 8
    def seg(start, end):
        c = sum(clicks[start:end])
        p = sum(purch[start:end])
        return (p / c) if c > 0 else 0.0
    # compute drift last-7 vs prior-7
    if end_idx - 14 >= 0:
        cvr_last7 = seg(end_idx - 7, end_idx)
        cvr_prev7 = seg(end_idx - 14, end_idx - 7)
        drift = abs(cvr_last7 - cvr_prev7) / max(1e-6, cvr_prev7) if cvr_prev7 > 0 else 1.0
    else:
        drift = 0.0
    # counts in 21-day window
    start21 = max(0, end_idx - 21)
    clicks21 = sum(clicks[start21:end_idx])
    # Rule:
    # - If drift > 0.25 or clicks21 > 20000, pick 7-day (more reactive)
    # - Else if clicks21 < 2000 and drift < 0.10, pick 21-day (more support)
    # - Else 14-day
    if drift > 0.25 or clicks21 > 20000:
        train = set(ds[end_idx - 7:end_idx])
    elif clicks21 < 2000 and drift < 0.10:
        train = set(ds[end_idx - 21:end_idx])
    else:
        train = set(ds[end_idx - 14:end_idx])
    test = set(ds[end_idx:end_idx + 7])
    return train, test


def _ewma_weights(dates: List[str], train_set: set, half_life_days: float = 7.0) -> Dict[str, float]:
    # weight by recency within train
    sorted_train = sorted(train_set)
    if not sorted_train:
        return {}
    t0 = datetime.strptime(sorted_train[-1], "%Y-%m-%d")
    lam = math.log(2) / max(1e-6, half_life_days)
    w = {}
    for d in sorted_train:
        dt = datetime.strptime(d, "%Y-%m-%d")
        days = (t0 - dt).days
        w[d] = math.exp(-lam * days)
    # normalize to mean 1.0
    m = np.mean(list(w.values())) if w else 1.0
    for k in list(w.keys()):
        w[k] = w[k] / m
    return w


def _simulate_day(spend: float, mu: float, sigma: float, cpc_lo: float, cpc_hi: float,
                  alpha: float, beta: float, wd_mult: float, freq_effect: float,
                  draws: int = 300,
                  mu2: float | None = None, sigma2: float | None = None, tail_w: float = 0.0) -> Tuple[float, float, float]:
    spend = max(0.0, float(spend))
    if spend <= 0:
        return 0.0, 0.0, 0.0
    # sample CPCs; allow a simple two-component mixture to capture heavy tails
    if (mu2 is not None) and (sigma2 is not None) and (tail_w > 0.0):
        comp2 = (np.random.rand(draws) < tail_w)
        cpcs = np.where(comp2,
                        np.random.lognormal(mu2, sigma2, size=draws),
                        np.random.lognormal(mu, sigma, size=draws))
    else:
        cpcs = np.random.lognormal(mu, sigma, size=draws)
    cpcs = np.clip(cpcs, max(1e-3, cpc_lo), max(cpc_hi, cpc_lo + 1e-3))
    lam_clicks = spend / cpcs
    max_clicks_cap = np.minimum(50000, (spend / np.maximum(0.1, cpc_lo)) + 1000)
    clicks = np.random.poisson(lam=np.minimum(lam_clicks, max_clicks_cap))
    p_cvr = np.random.beta(alpha, beta, size=draws) * wd_mult * freq_effect
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

    # choose train/test per campaign, compute priors
    acc_clicks = 0.0
    acc_purch = 0.0
    all_cpcs_train: List[float] = []
    windows: Dict[str, Tuple[set, set]] = {}
    for cid, rows in by_camp.items():
        ds = [r["date"] for r in rows]
        clicks = [r["clicks"] for r in rows]
        purch = [r["purch"] for r in rows]
        train_set, test_set = _auto_window(ds, clicks, purch)
        windows[cid] = (train_set, test_set)
        for r in rows:
            if r["date"] in train_set:
                acc_clicks += r["clicks"]
                acc_purch += r["purch"]
                if r["clicks"] > 0 and r["spend"] > 0:
                    all_cpcs_train.append(r["spend"]/r["clicks"])
    a0, b0 = _beta_hyper_from_global(acc_purch, acc_clicks, k=150.0)
    g_mu, g_sigma, g_p5, g_p95 = _fit_lognorm(all_cpcs_train or [2.0,2.2,2.5])

    # weekday multipliers (account-level) for modest temporal effect reuse
    # we’ll reuse the Phase-2 method quickly: ratio per weekday vs account average
    acc_wd_counts = {w: {"succ": 0.0, "tot": 0.0} for w in range(7)}
    for cid, rows in by_camp.items():
        train_set, _ = windows[cid]
        for r in rows:
            if r["date"] in train_set:
                w = _weekday(r["date"]) 
                acc_wd_counts[w]["succ"] += r["purch"]
                acc_wd_counts[w]["tot"] += max(r["clicks"], 0)
    acc_rate = (acc_purch/acc_clicks) if acc_clicks>0 else 0.01
    acc_wd_rate = {}
    for w in range(7):
        tot = acc_wd_counts[w]["tot"]
        succ = acc_wd_counts[w]["succ"]
        r = (succ/tot) if tot>0 else acc_rate
        acc_wd_rate[w] = r/max(acc_rate,1e-6)

    params = {}
    wd_mults = {}
    freq_slope = {}
    med_freq = {}

    for cid, rows in by_camp.items():
        train_set, test_set = windows[cid]
        # CPC fit on train
        cpcs = [r["spend"]/r["clicks"] for r in rows if r["date"] in train_set and r["spend"]>0 and r["clicks"]>0]
        mu,sigma,p5,p95 = _fit_lognorm(cpcs) if len(cpcs)>=3 else (g_mu,g_sigma,g_p5,g_p95)
        # Estimate tail weight from empirical > p90 mass
        if len(cpcs) >= 10:
            p90_emp = float(np.percentile(np.array(cpcs), 90))
            tail_frac = float(np.mean(np.array(cpcs) > p90_emp))
        else:
            tail_frac = 0.1
        tail_w = float(np.clip(tail_frac, 0.05, 0.4))
        # time-decayed clicks/purch for Beta prior
        weights = _ewma_weights([r["date"] for r in rows], train_set, half_life_days=float(os.getenv("AELP2_DECAY_HL", "7")))
        w_clicks = 0.0
        w_purch = 0.0
        for r in rows:
            d=r["date"]
            if d in train_set:
                w = weights.get(d, 1.0)
                w_clicks += w * r["clicks"]
                w_purch += w * r["purch"]
        alpha = a0 + w_purch
        beta = b0 + max(0.0, w_clicks - w_purch)
        # weekday mults: shrink to account-level
        wd_counts = {w: {"succ": 0.0, "tot": 0.0} for w in range(7)}
        for r in rows:
            if r["date"] in train_set:
                w=_weekday(r["date"])
                wd_counts[w]["succ"] += r["purch"]
                wd_counts[w]["tot"] += max(r["clicks"], 0)
        base = (w_purch/w_clicks) if w_clicks>0 else acc_rate
        wmult = {}
        for w in range(7):
            tot=wd_counts[w]["tot"]
            succ=wd_counts[w]["succ"]
            local=(succ/tot) if tot>0 else base
            ratio_local= local/max(base,1e-6)
            weight=min(1.0, tot/500.0)
            ratio= weight*ratio_local + (1-weight)*acc_wd_rate[w]
            wmult[w] = float(np.clip(ratio, 0.7, 1.3))
        wd_mults[cid] = wmult
        # frequency slope (as in Phase 2)
        freqs = [r["freq"] for r in rows if r["date"] in train_set and r["clicks"]>0]
        m_f = float(np.median(freqs)) if freqs else 1.5
        xs, ys = [] , []
        cvrs_pos = [ (rr["purch"]/rr["clicks"]) for rr in rows if rr["date"] in train_set and rr["clicks"]>0 and (rr["purch"]/rr["clicks"])>0 ]
        eps = max(1e-6, float(np.percentile(np.array(cvrs_pos), 5)) if len(cvrs_pos)>=1 else 1e-5)
        for r in rows:
            if r["date"] not in train_set or r["clicks"]<=0: continue
            cvr = r["purch"]/r["clicks"]
            xs.append(r["freq"] - m_f)
            ys.append(math.log(max(cvr,0)+eps))
        if len(xs)>=5:
            X=np.vstack([np.ones(len(xs)), np.array(xs)]).T
            coef,*_=np.linalg.lstsq(X, np.array(ys), rcond=None)
            slope=float(coef[1])
        else:
            slope=0.0
        slope=float(np.clip(slope, -0.25, 0.05))
        freq_slope[cid]=slope
        med_freq[cid]=m_f
        # Creative age: derive median created_time across campaign ads
        try:
            ad_times = _get_campaign_creatives_created(token, cid, max_pages=3)
            if ad_times:
                ad_days = sorted([datetime.strptime(t, "%Y-%m-%d").date() for t in ad_times])
                median_ct = ad_days[len(ad_days)//2]
            else:
                median_ct = (today - timedelta(days=14))
        except Exception:
            median_ct = (today - timedelta(days=14))
        for r in rows:
            rd = datetime.strptime(r["date"], "%Y-%m-%d").date()
            r["cage"] = max(0, (rd - median_ct).days)
        ages_train = [r["cage"] for r in rows if r["date"] in train_set and r["clicks"]>0]
        m_a = float(np.median(ages_train)) if ages_train else 7.0
        # regress log-cvr on (age - median)
        xa, ya = [], []
        for r in rows:
            if r["date"] not in train_set or r["clicks"]<=0: continue
            cvr = r["purch"]/r["clicks"]
            xa.append(r["cage"] - m_a)
            ya.append(math.log(max(cvr,0)+eps))
        if len(xa) >= 5:
            Xa = np.vstack([np.ones(len(xa)), np.array(xa)]).T
            coef,*_ = np.linalg.lstsq(Xa, np.array(ya), rcond=None)
            a_slope = float(coef[1])
        else:
            a_slope = 0.0
        a_slope = float(np.clip(a_slope, -0.15, 0.02))
        mu2 = mu + 0.3 * sigma
        sigma2 = min(2.5, sigma * 1.8)
        params[cid] = {"mu":mu,"sigma":sigma,
                       "mu2":mu2, "sigma2":sigma2, "tail_w":tail_w,
                       "cpc_p5":p5,"cpc_p95":p95,
                       "alpha":alpha,"beta":beta,
                       "age_med": m_a, "age_slope": a_slope,
                       "name": rows[0]["name"] if rows else cid}

    # simulate per day in each campaign’s test window and aggregate
    # Align test days to the last 7 days in the dataset for account-level comparability
    all_ds = sorted(set([r["date"] for rows in by_camp.values() for r in rows]))
    days_sorted = all_ds[-7:]
    day_actual = {d: {"spend": 0.0, "purch": 0.0} for d in days_sorted}
    day_pred = {d: {"med": 0.0, "p10": 0.0, "p90": 0.0} for d in days_sorted}
    for cid, rows in by_camp.items():
        tr, te = windows[cid]
        p = params[cid]
        for r in rows:
            d=r["date"]
            if d not in te or d not in day_actual: continue
            spend=r["spend"]; day_actual[d]["spend"]+=spend; day_actual[d]["purch"]+=r["purch"]
            w=_weekday(d); wd = wd_mults[cid][w]
            f=r["freq"]; eff=float(np.clip(math.exp(freq_slope[cid]*(f-med_freq[cid])),0.7,1.2))
            # add creative-age effect around campaign median age
            a = r.get("cage", 7.0)
            a_med = p.get("age_med", 7.0)
            a_slope = p.get("age_slope", 0.0)
            age_eff = float(np.clip(math.exp(a_slope*(a - a_med)), 0.7, 1.2))
            eff *= age_eff
            med,p10,p90=_simulate_day(
                spend,
                p["mu"], p["sigma"], p["cpc_p5"], p["cpc_p95"],
                p["alpha"], p["beta"], wd, eff,
                draws=300,
                mu2=p.get("mu2"), sigma2=p.get("sigma2"), tail_w=p.get("tail_w", 0.0)
            )
            day_pred[d]["med"]+=med; day_pred[d]["p10"]+=p10; day_pred[d]["p90"]+=p90

    # Calibrate intervals (shrink toward median) to target narrower 80% bands
    gamma = float(os.getenv("AELP2_PI_SHRINK", "0.8"))
    if gamma < 0.2 or gamma > 1.0:
        gamma = 0.8
    for d in days_sorted:
        p = day_pred[d]
        med = p["med"]
        p["p10"] = med - gamma * (med - p["p10"]) if p["p10"] <= med else med
        p["p90"] = med + gamma * (p["p90"] - med) if p["p90"] >= med else med

    purch_mape, cac_mape, cov_hits, cov_tot = [], [], 0, 0
    daily_rows = []
    for d in days_sorted:
        a=day_actual[d]; p=day_pred[d]
        spend=float(a["spend"]); act=float(a["purch"]); pred=max(0.0,float(p["med"]))
        if act>0:
            purch_mape.append(abs(pred-act)/act)
        act_cac=spend/act if act>0 else float("inf")
        pred_cac=spend/pred if pred>0 else float("inf")
        if math.isfinite(act_cac) and math.isfinite(pred_cac) and act_cac>0:
            cac_mape.append(abs(pred_cac-act_cac)/act_cac)
        lo,hi=float(p["p10"]),float(p["p90"])
        if act>=lo and act<=hi: cov_hits+=1
        cov_tot+=1
        daily_rows.append({"date": d, "spend": round(spend,2), "actual_purch": round(act,2), "pred_purch_med": round(pred,2), "pred_purch_p10": round(lo,2), "pred_purch_p90": round(hi,2)})

    out={
        "summary": {
            "purchases_day_mape": (round(float(np.mean(purch_mape)*100),2) if purch_mape else None),
            "cac_day_mape": (round(float(np.mean(cac_mape)*100),2) if cac_mape else None),
            "coverage80": (round(100.0*cov_hits/cov_tot,1) if cov_tot else None),
        },
        "daily": daily_rows,
        "note": "v2 uses per-campaign auto windows and time-decayed priors (half-life env AELP2_DECAY_HL)"
    }

    os.makedirs("AELP2/reports", exist_ok=True)
    path="AELP2/reports/sim_fidelity_campaigns_temporal_v2.json"
    with open(path,"w",encoding="utf-8") as f:
        json.dump(out,f,indent=2)
    print(path)


if __name__ == "__main__":
    np.random.seed(int(os.getenv("AELP2_SEED","47")))
    main()
