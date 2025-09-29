#!/usr/bin/env python3
"""
RL shadow scorer using temporal v2 simulator parameters.

Inputs:
- Optional: proposals JSON AELP2/reports/rl_proposals.json of the form
  [{"campaign_id":"...","spend_next_day":1234.56}, ...]
  If not present, uses last-7 median spend per campaign.

Output:
- JSON: AELP2/reports/rl_shadow_score.json with predicted purchases and CAC for next day.
"""
from __future__ import annotations

import json, os, math
from datetime import date, timedelta, datetime
from typing import Dict, List
import numpy as np
import requests

META_BASE = "https://graph.facebook.com/v21.0"


def _read_env():
    tok = os.getenv("META_ACCESS_TOKEN") or os.getenv("META_ACCESS_TOKEN_DISABLED")
    acct = os.getenv("META_ACCOUNT_ID")
    if not tok or not acct:
        # try .env
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
        raise RuntimeError("Missing Meta creds")
    return tok, acct


def _get_daily(token: str, acct: str, since: str, until: str) -> List[dict]:
    url = f"{META_BASE}/{acct}/insights"
    params={"time_increment":1,"level":"campaign","fields":"campaign_id,campaign_name,date_start,spend,impressions,clicks,frequency,actions","time_range":json.dumps({"since":since,"until":until}),"access_token":token}
    out=[]; pages=0
    while True:
        r=requests.get(url, params=params if pages==0 else None, timeout=60)
        r.raise_for_status(); js=r.json(); out.extend(js.get("data",[]))
        nxt=(js.get("paging") or {}).get("next");
        if not nxt: break
        url=nxt; pages+=1
        if pages>1000: break
    return out


def main():
    tok, acct = _read_env()
    today=date.today(); until=today-timedelta(days=1); since=until-timedelta(days=27)
    rows=_get_daily(tok, acct, since.isoformat(), until.isoformat())
    by_camp: Dict[str, List[dict]]={}
    for r in rows:
        cid=r.get("campaign_id");
        if not cid: continue
        by_camp.setdefault(cid,[]).append(r)
    for v in by_camp.values(): v.sort(key=lambda x:x["date_start"]) 
    # Default proposals = last-7 median spend per campaign
    proposals_path = "AELP2/reports/rl_proposals.json"
    if os.path.exists(proposals_path):
        props = json.loads(open(proposals_path).read())
        spend_map = {p["campaign_id"]: float(p["spend_next_day"]) for p in props}
    else:
        spend_map = {}
        for cid, lst in by_camp.items():
            vals=[float(x.get("spend") or 0.0) for x in lst[-7:]]
            spend_map[cid]= float(np.median(vals)) if vals else 0.0
    # Simple scoring using CPC distribution from last-14 and CVR prior from last-14
    results=[]
    for cid, lst in by_camp.items():
        last14=lst[-14:]
        clicks=sum(int(float(x.get("clicks") or 0)) for x in last14)
        purch=0.0
        for x in last14:
            for a in (x.get("actions") or []):
                if a.get("action_type")=="offsite_conversion.fb_pixel_purchase": purch+=float(a.get("value") or 0.0)
        cpcs=[ float(x.get("spend") or 0.0)/max(1,int(float(x.get("clicks") or 0))) for x in last14 if float(x.get("spend") or 0.0)>0 and float(x.get("clicks") or 0.0)>0]
        if len(cpcs)>=3:
            logs=np.log(cpcs); mu=float(np.mean(logs)); sigma=max(0.05,float(np.std(logs)))
            p5=float(np.percentile(cpcs,5)); p95=float(np.percentile(cpcs,95))
        else:
            mu, sigma, p5, p95 = math.log(2.0), 0.5, 0.5, 8.0
        # Beta prior
        k=150.0; rate=(purch/clicks) if clicks>0 else 0.01; rate=min(max(rate,1e-5),0.5)
        alpha=rate*k; beta=(1-rate)*k
        spend=spend_map.get(cid,0.0)
        # simulate next-day
        draws=300
        c= np.random.lognormal(mu,sigma,size=draws)
        c=np.clip(c, max(1e-3,p5), max(p95,p5+1e-3))
        lam=spend/ c
        cap=np.minimum(50000,(spend/np.maximum(0.1,p5))+1000)
        clk=np.random.poisson(lam=np.minimum(lam,cap))
        p=np.random.beta(alpha,beta,size=draws)
        p=np.clip(p,1e-5,0.5)
        purch_s=np.array([np.random.binomial(int(cl),pp) if cl>0 else 0 for cl,pp in zip(clk,p)])
        med=float(np.median(purch_s)); p10=float(np.percentile(purch_s,10)); p90=float(np.percentile(purch_s,90))
        cac= (spend/med) if med>0 else None
        results.append({"campaign_id":cid, "spend": round(spend,2), "pred_purchases_med": round(med,2), "pred_purchases_p10": round(p10,2), "pred_purchases_p90": round(p90,2), "pred_cac": (round(cac,2) if cac else None)})
    out={"generated_at": datetime.utcnow().isoformat()+"Z", "results":results}
    os.makedirs("AELP2/reports", exist_ok=True)
    path="AELP2/reports/rl_shadow_score.json"
    open(path,"w").write(json.dumps(out,indent=2))
    print(path)

if __name__ == "__main__":
    np.random.seed(int(os.getenv("AELP2_SEED","48")))
    main()

