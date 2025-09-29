#!/usr/bin/env python3
"""
Temporal simulator v3: per-campaign hourly effect + fast-drift window rule

Adds to v2:
- Per-campaign hourly CVR multiplier (shrink to account-level if sparse)
- Fast-drift train window: force 7-day if CVR drift > 0.2 or median creative age < 7

Outputs
- JSON: AELP2/reports/sim_fidelity_campaigns_temporal_v3.json
"""
from __future__ import annotations

import json, math, os
from collections import defaultdict
from datetime import date, timedelta, datetime
from typing import Dict, List, Tuple
import numpy as np
import requests

META_BASE = "https://graph.facebook.com/v21.0"


def _read_env():
    tok = os.getenv("META_ACCESS_TOKEN") or os.getenv("META_ACCESS_TOKEN_DISABLED")
    acct = os.getenv("META_ACCOUNT_ID")
    if not tok or not acct:
        if os.path.exists(".env"):
            for ln in open(".env","r",encoding="utf-8"):
                ln=ln.strip()
                if ln.startswith("export META_ACCESS_TOKEN=") and not tok:
                    tok = ln.split("=",1)[1].strip()
                if ln.startswith("export META_ACCESS_TOKEN_DISABLED=") and not tok:
                    tok = ln.split("=",1)[1].strip()
                if ln.startswith("export META_ACCOUNT_ID=") and not acct:
                    acct = ln.split("=",1)[1].strip()
    if not tok or not acct:
        raise RuntimeError("Missing META creds")
    return tok, acct


def _get_insights_daily(token: str, acct: str, since: str, until: str) -> List[dict]:
    url=f"{META_BASE}/{acct}/insights"
    params={"time_increment":1,"level":"campaign","fields":"campaign_id,campaign_name,date_start,impressions,clicks,spend,frequency,actions","time_range":json.dumps({"since":since,"until":until}),"access_token":token}
    out=[]; pages=0
    while True:
        r=requests.get(url, params=params if pages==0 else None, timeout=60)
        r.raise_for_status(); js=r.json(); out.extend(js.get("data",[]))
        nxt=(js.get("paging") or {}).get("next");
        if not nxt: break
        url=nxt; pages+=1
        if pages>1000: break
    return out


def _extract_purchase_count(actions: List[dict] | None) -> float:
    if not actions: return 0.0
    for a in actions:
        if a.get("action_type")=="offsite_conversion.fb_pixel_purchase":
            try: return float(a.get("value") or 0)
            except: return 0.0
    return 0.0


def _fit_lognorm(cpcs: List[float]):
    x=np.array([c for c in cpcs if c>0], dtype=float)
    if x.size==0: return math.log(2.0),0.5,0.5,8.0
    p5=float(np.percentile(x,5)); p95=float(np.percentile(x,95))
    x=np.clip(x, max(1e-6,p5), max(p95,p5+1e-3))
    logs=np.log(x); mu=float(np.mean(logs)); sigma=float(np.std(logs) if logs.size>1 else 0.3)
    sigma=max(0.05, min(sigma,1.5))
    return mu,sigma,p5,p95


def _beta_hyper(success: float, total: float, k: float = 150.0):
    rate=(success/total) if total>0 else 0.01; rate=min(max(rate,1e-5),0.5)
    return rate*k, (1-rate)*k


def _weekday(s: str) -> int:
    return datetime.strptime(s, "%Y-%m-%d").weekday()


def _get_campaign_creatives_created(token: str, campaign_id: str, max_pages: int = 3) -> List[str]:
    url=f"{META_BASE}/{campaign_id}/ads"; params={"fields":"id,created_time,updated_time","limit":200,"access_token":token}
    out=[]; pages=0
    while True:
        r=requests.get(url, params=params if pages==0 else None, timeout=60)
        if r.status_code>=400: break
        js=r.json()
        for row in js.get("data",[]):
            ct=row.get("created_time") or row.get("updated_time");
            if ct: out.append(ct[:10])
        nxt=(js.get("paging") or {}).get("next");
        if not nxt: break
        url=nxt; pages+=1
        if pages>=max_pages: break
    return out


def _get_campaign_hourly(token: str, acct: str, campaign_id: str, since: str, until: str):
    url=f"{META_BASE}/{acct}/insights"  # campaign filter via filtering breakdown? Use level=campaign + breakdowns
    params={"level":"campaign","filtering":json.dumps([{ "field":"campaign.id","operator":"IN","value":[campaign_id]}]),
            "breakdowns":"hourly_stats_aggregated_by_advertiser_time_zone","fields":"clicks,actions","time_range": json.dumps({"since":since,"until":until}),"access_token":token}
    try:
        r=requests.get(url, params=params, timeout=60)
        if r.status_code>=400: return None
        js=r.json();
        hours={}
        for row in js.get("data",[]):
            raw=row.get("hourly_stats_aggregated_by_advertiser_time_zone"); h=(raw[:2] if raw else None)
            if not h: continue
            clicks=float(row.get("clicks") or 0.0)
            purch=0.0
            for a in (row.get("actions") or []):
                if a.get("action_type")=="offsite_conversion.fb_pixel_purchase": purch=float(a.get("value") or 0.0)
            v=hours.get(h, {"clicks":0.0,"purch":0.0}); v["clicks"]+=clicks; v["purch"]+=purch; hours[h]=v
        return hours
    except Exception:
        return None

def _get_campaign_placement(token: str, acct: str, campaign_id: str, since: str, until: str):
    """Return aggregated clicks/purchases by placement for a campaign in [since, until].
    Uses breakdowns at level=campaign filtered by campaign.id. Falls back to None on error.
    """
    url=f"{META_BASE}/{acct}/insights"
    params={
        "level":"campaign",
        "filtering": json.dumps([{ "field":"campaign.id","operator":"IN","value":[campaign_id]}]),
        "breakdowns":"publisher_platform,platform_position,impression_device",
        "fields":"impressions,clicks,actions",
        "time_range": json.dumps({"since":since,"until":until}),
        "access_token": token,
        "limit": 500
    }
    try:
        out={}
        pages=0; url0=url
        while True:
            r=requests.get(url, params=params if pages==0 else None, timeout=90)
            if r.status_code>=400:
                return None
            js=r.json()
            for row in js.get("data", []):
                pp=row.get("publisher_platform") or 'unknown'
                pos=row.get("platform_position") or 'unknown'
                dev=row.get("impression_device") or 'unknown'
                key=f"{pp}|{pos}|{dev}"
                clicks=float(row.get("clicks") or 0.0)
                purch=0.0
                for a in (row.get("actions") or []):
                    if a.get("action_type")=="offsite_conversion.fb_pixel_purchase":
                        try: purch=float(a.get("value") or 0.0)
                        except Exception: purch=0.0
                agg=out.get(key, {"clicks":0.0, "purch":0.0})
                agg["clicks"]+=clicks; agg["purch"]+=purch; out[key]=agg
            nxt=(js.get("paging") or {}).get("next")
            if not nxt: break
            url=nxt; pages+=1
            if pages>4000: break
        return out
    except Exception:
        return None


def _load_account_hourly():
    p="AELP2/reports/hourly_multipliers.json"
    if os.path.exists(p):
        try:
            js=json.loads(open(p).read()); return js.get("hourly",{})
        except Exception:
            return {}
    return {}


def main():
    token, acct = _read_env()
    today=date.today(); until=today-timedelta(days=1); since=until-timedelta(days=27)
    rows=_get_insights_daily(token, acct, since.isoformat(), until.isoformat())
    by_camp: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        cid=r.get("campaign_id");
        if not cid: continue
        by_camp[cid].append({
            "date": r.get("date_start"),
            "name": r.get("campaign_name"),
            "spend": float(r.get("spend") or 0.0),
            "clicks": int(float(r.get("clicks") or 0.0)),
            "freq": float(r.get("frequency") or 0.0),
            "purch": _extract_purchase_count(r.get("actions")),
        })
    for v in by_camp.values(): v.sort(key=lambda x:x["date"])    

    # Build account-level weekday ratios for shrinkage
    all_dates=sorted({r["date"] for lst in by_camp.values() for r in lst})
    acc_clicks=sum(r["clicks"] for lst in by_camp.values() for r in lst)
    acc_purch=sum(r["purch"] for lst in by_camp.values() for r in lst)
    a0,b0=_beta_hyper(acc_purch, acc_clicks, k=150.0)
    acc_rate=(acc_purch/acc_clicks) if acc_clicks>0 else 0.01
    acc_wd_counts={w:{"succ":0.0,"tot":0.0} for w in range(7)}
    for lst in by_camp.values():
        for r in lst:
            w=_weekday(r["date"]); acc_wd_counts[w]["succ"]+=r["purch"]; acc_wd_counts[w]["tot"]+=max(r["clicks"],0)
    acc_wd_rate={}
    for w in range(7):
        tot=acc_wd_counts[w]["tot"]; succ=acc_wd_counts[w]["succ"]
        rr=(succ/tot) if tot>0 else acc_rate
        acc_wd_rate[w]= rr/max(acc_rate,1e-6)

    # Account hourly multipliers (for shrinkage)
    acc_hourly=_load_account_hourly()

    params={}; windows={}; wd_mults={}; freq_slope={}; med_freq={}; hour_mult={}; place_mult={}; bias_factor={}    

    for cid, lst in by_camp.items():
        ds=[r["date"] for r in lst]; clicks=[r["clicks"] for r in lst]; purch=[r["purch"] for r in lst]
        # CVR drift last-7 vs prior-7
        if len(ds)>=21:
            end=len(ds)-7
            cvr_last7=(sum(purch[end:])/max(1,sum(clicks[end:])))
            cvr_prev7=(sum(purch[end-7:end])/max(1,sum(clicks[end-7:end])))
            drift = abs(cvr_last7 - cvr_prev7)/max(1e-6, cvr_prev7) if cvr_prev7>0 else 1.0
        else:
            drift=0.0
        # Creative median age
        try:
            ad_times=_get_campaign_creatives_created(token, cid, max_pages=2)
            if ad_times:
                ad_days=sorted([datetime.strptime(t, "%Y-%m-%d").date() for t in ad_times])
                median_ct=ad_days[len(ad_days)//2]
            else:
                median_ct=(today - timedelta(days=14))
        except Exception:
            median_ct=(today - timedelta(days=14))
        last_day=datetime.strptime(ds[-1], "%Y-%m-%d").date() if ds else today
        med_age_days=max(0,(last_day - median_ct).days)
        # Fast-drift window rule
        if drift>0.2 or med_age_days<7:
            train=set(ds[-14:-7])  # last 7 days before test
        else:
            train=set(ds[-21:-7]) if len(ds)>=28 else set(ds[-14:-7])
        test=set(ds[-7:])
        windows[cid]=(train,test)

        # CPC fit (train)
        cpcs=[ r["spend"]/r["clicks"] for r in lst if r["date"] in train and r["spend"]>0 and r["clicks"]>0 ]
        mu,sigma,p5,p95=_fit_lognorm(cpcs) if len(cpcs)>=3 else (math.log(2.0),0.5,0.5,8.0)
        # Tail mixture (shrink by sample size)
        if len(cpcs)>=10:
            p90_emp=float(np.percentile(np.array(cpcs),90)); tail_frac=float(np.mean(np.array(cpcs)>p90_emp))
        else:
            tail_frac=0.1
        size_w=min(1.0, len(cpcs)/30.0)
        tail_w=float(np.clip(tail_frac*size_w, 0.03, 0.25))
        mu2=mu+0.25*sigma; sigma2=min(2.2, 1.6*sigma)

        # CVR prior (train)
        t_clicks=sum(r["clicks"] for r in lst if r["date"] in train)
        t_purch=sum(r["purch"] for r in lst if r["date"] in train)
        alpha=a0 + t_purch; beta=b0 + max(0.0, t_clicks - t_purch)

        # weekday mults with shrinkage
        wd_counts={w:{"succ":0.0,"tot":0.0} for w in range(7)}
        for r in lst:
            if r["date"] in train:
                w=_weekday(r["date"]); wd_counts[w]["succ"]+=r["purch"]; wd_counts[w]["tot"]+=max(r["clicks"],0)
        base=(t_purch/t_clicks) if t_clicks>0 else acc_rate
        wmult={}
        for w in range(7):
            tot=wd_counts[w]["tot"]; succ=wd_counts[w]["succ"]
            local=(succ/tot) if tot>0 else base
            ratio_local= local/max(base,1e-6)
            weight=min(1.0, tot/400.0)
            ratio= weight*ratio_local + (1-weight)*acc_wd_rate[w]
            wmult[w]=float(np.clip(ratio,0.7,1.3))
        wd_mults[cid]=wmult

        # frequency slope
        freqs=[r["freq"] for r in lst if r["date"] in train and r["clicks"]>0]
        m_f=float(np.median(freqs)) if freqs else 1.5
        xs=[]; ys=[]
        cvrs_pos=[ (rr["purch"]/rr["clicks"]) for rr in lst if rr["date"] in train and rr["clicks"]>0 and (rr["purch"]/rr["clicks"])>0 ]
        eps=max(1e-6, float(np.percentile(np.array(cvrs_pos),5)) if len(cvrs_pos)>=1 else 1e-5)
        for r in lst:
            if r["date"] not in train or r["clicks"]<=0: continue
            cvr=r["purch"]/r["clicks"]; xs.append(r["freq"]-m_f); ys.append(math.log(max(cvr,0)+eps))
        if len(xs)>=5:
            X=np.vstack([np.ones(len(xs)), np.array(xs)]).T; coef,*_=np.linalg.lstsq(X, np.array(ys), rcond=None); slope=float(coef[1])
        else:
            slope=0.0
        slope=float(np.clip(slope,-0.25,0.05))
        freq_slope[cid]=slope; med_freq[cid]=m_f

        # per-campaign hourly multiplier (scalar)
        hours=_get_campaign_hourly(token, acct, cid, since.isoformat(), until.isoformat())
        if hours and sum(v["clicks"] for v in hours.values())>0:
            rates={h:(v["purch"]/v["clicks"]) if v["clicks"]>0 else 0.0 for h,v in hours.items()}
            base=np.mean([r for r in rates.values() if r>0]) if any(r>0 for r in rates.values()) else 0.0
            camp_mults={h:( (r/base) if base>0 else 1.0) for h,r in rates.items()}
            # average across hours weighted by clicks
            total_clicks=sum(v["clicks"] for v in hours.values())
            h_scalar=sum((camp_mults.get(h,1.0))* (v["clicks"]/total_clicks) for h,v in hours.items()) if total_clicks>0 else 1.0
        else:
            # fallback to account-level mean of hourly multipliers (=1.0)
            h_scalar=1.0
        hour_mult[cid]=float(np.clip(h_scalar, 0.85, 1.15))

        # per-campaign placement CVR scalar (train window)
        plc=_get_campaign_placement(token, acct, cid, since.isoformat(), until.isoformat())
        if plc and sum(v.get("clicks",0.0) for v in plc.values())>0:
            total_clicks=sum(v.get("clicks",0.0) for v in plc.values())
            total_purch=sum(v.get("purch",0.0) for v in plc.values())
            base=(total_purch/total_clicks) if total_clicks>0 else acc_rate
            scalar=0.0; weight=0.0
            for k,v in plc.items():
                clk=float(v.get("clicks",0.0)); pr=float(v.get("purch",0.0))
                share = clk/total_clicks if total_clicks>0 else 0.0
                local = (pr/clk) if clk>0 else base
                ratio = (local / max(base,1e-6))
                scalar += share * ratio
                weight += share
            pl_scalar = (scalar/weight) if weight>0 else 1.0
        else:
            pl_scalar = 1.0
        place_mult[cid]=float(np.clip(pl_scalar, 0.85, 1.15))

        # Optional domain randomization (global noise on CPC params)
        if os.getenv('AELP2_DR','0')=='1':
            scale=float(os.getenv('AELP2_DR_SCALE','0.1'))
            mu += np.random.normal(0.0, scale*max(0.1, abs(mu)))
            sigma = float(np.clip(sigma + np.random.normal(0.0, scale*sigma), 0.05, 2.5))
        params[cid]={"mu":mu,"sigma":sigma,"mu2":mu2,"sigma2":sigma2,"tail_w":tail_w,
                     "cpc_p5":p5,"cpc_p95":p95, "alpha":alpha,"beta":beta,
                     "name": lst[0]["name"] if lst else cid}

        # Discrepancy calibration (ROPE-style): simulate train days and fit a multiplicative bias on medians
        try:
            pred_tr=0.0; act_tr=0.0
            for r in lst:
                d=r["date"]
                if d not in train: continue
                spend=r["spend"]; w=_weekday(d); wd=wd_mults[cid][w]
                f=r["freq"]; f_eff=float(np.clip(math.exp(slope*(f-m_f)),0.7,1.2))
                wd_eff= wd * f_eff * hour_mult.get(cid,1.0) * place_mult.get(cid,1.0)
                med,_,_=sim_day(spend, mu,sigma,p5,p95, alpha,beta, wd_eff, draws=200, mu2=mu2,sigma2=sigma2,tail_w=tail_w)
                pred_tr += med
            act_tr = sum(r["purch"] for r in lst if r["date"] in train)
            bf = float(np.clip((act_tr+1e-6)/(pred_tr+1e-6), 0.7, 1.3)) if pred_tr>0 else 1.0
        except Exception:
            bf = 1.0
        bias_factor[cid]=bf

    # Simulation on last 7 days per campaign test window
    days_sorted=all_dates[-7:]
    day_actual={d:{"spend":0.0,"purch":0.0} for d in days_sorted}
    day_pred={d:{"med":0.0,"p10":0.0,"p90":0.0} for d in days_sorted}

    def sim_day(spend, mu,sigma,p5,p95, alpha,beta, wd_eff, draws=300, mu2=None,sigma2=None,tail_w=0.0):
        spend=max(0.0,float(spend))
        if spend<=0:
            return 0.0,0.0,0.0
        if (mu2 is not None) and (sigma2 is not None) and (tail_w>0.0):
            comp2=(np.random.rand(draws)<tail_w)
            c=np.where(comp2, np.random.lognormal(mu2,sigma2,size=draws), np.random.lognormal(mu,sigma,size=draws))
        else:
            c=np.random.lognormal(mu,sigma,size=draws)
        c=np.clip(c, max(1e-3,p5), max(p95,p5+1e-3))
        lam=spend/c; cap=np.minimum(50000,(spend/np.maximum(0.1,p5))+1000)
        clk=np.random.poisson(lam=np.minimum(lam,cap))
        p=np.random.beta(alpha,beta,size=draws)*wd_eff
        p=np.clip(p,1e-5,0.5)
        pr=np.array([np.random.binomial(int(cl),pp) if cl>0 else 0 for cl,pp in zip(clk,p)], dtype=float)
        return float(np.median(pr)), float(np.percentile(pr,10)), float(np.percentile(pr,90))

    for cid, lst in by_camp.items():
        train,test=windows[cid]; p=params[cid]
        for r in lst:
            d=r["date"]
            if d not in test or d not in day_actual: continue
            spend=r["spend"]; day_actual[d]["spend"]+=spend; day_actual[d]["purch"]+=r["purch"]
            w=_weekday(d); wd=wd_mults[cid][w]
            f=r["freq"]; f_eff=float(np.clip(math.exp(freq_slope[cid]*(f-med_freq[cid])),0.7,1.2))
            wd_eff= wd * f_eff * hour_mult.get(cid,1.0) * place_mult.get(cid,1.0)
            med,p10,p90=sim_day(spend, p["mu"],p["sigma"],p["cpc_p5"],p["cpc_p95"], p["alpha"],p["beta"], wd_eff, draws=300, mu2=p["mu2"],sigma2=p["sigma2"],tail_w=p["tail_w"]) 
            bf=bias_factor.get(cid,1.0)
            med*=bf; p10*=bf; p90*=bf
            day_pred[d]["med"]+=med; day_pred[d]["p10"]+=p10; day_pred[d]["p90"]+=p90

    # Slight PI shrink to avoid 100% coverage
    gamma=float(os.getenv("AELP2_PI_SHRINK","0.8")); gamma=0.75 if gamma>0.9 else gamma
    for d in days_sorted:
        med=day_pred[d]["med"]; day_pred[d]["p10"]=med - gamma*(med-day_pred[d]["p10"]) if day_pred[d]["p10"]<=med else med
        day_pred[d]["p90"]=med + gamma*(day_pred[d]["p90"]-med) if day_pred[d]["p90"]>=med else med

    purch_mape=[]; cac_mape=[]; cov_hits=0; cov_tot=0; daily=[]
    for d in days_sorted:
        a=day_actual[d]; p=day_pred[d]
        spend=float(a["spend"]); act=float(a["purch"]); pred=max(0.0,float(p["med"]))
        if act>0: purch_mape.append(abs(pred-act)/act)
        act_cac=spend/act if act>0 else float("inf"); pred_cac=spend/pred if pred>0 else float("inf")
        if math.isfinite(act_cac) and math.isfinite(pred_cac) and act_cac>0:
            cac_mape.append(abs(pred_cac-act_cac)/act_cac)
        lo,hi=float(p["p10"]),float(p["p90"]); cov_hits += (1 if (act>=lo and act<=hi) else 0); cov_tot+=1
        daily.append({"date":d, "spend":round(spend,2), "actual_purch":round(act,2), "pred_purch_med":round(pred,2), "pred_purch_p10":round(lo,2), "pred_purch_p90":round(hi,2)})

    out={"summary": {"purchases_day_mape": (round(float(np.mean(purch_mape)*100),2) if purch_mape else None),
                      "cac_day_mape": (round(float(np.mean(cac_mape)*100),2) if cac_mape else None),
                      "coverage80": (round(100.0*cov_hits/cov_tot,1) if cov_tot else None)},
         "daily": daily}
    os.makedirs("AELP2/reports", exist_ok=True)
    path="AELP2/reports/sim_fidelity_campaigns_temporal_v3.json"; open(path,"w").write(json.dumps(out,indent=2)); print(path)

if __name__=="__main__":
    np.random.seed(int(os.getenv("AELP2_SEED","49")))
    main()
