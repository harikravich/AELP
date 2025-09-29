#!/usr/bin/env python3
"""
Fetch account-level hourly CVR multipliers (normalized to 1.0 mean).
Writes JSON: AELP2/reports/hourly_multipliers.json
Note: Useful for future hourly simulators; day-level sim remains neutral.
"""
from __future__ import annotations
import os, json
from datetime import date, timedelta
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
        raise RuntimeError("Missing Meta creds")
    return tok, acct


def main():
    tok, acct = _read_env()
    today = date.today(); until=today - timedelta(days=1); since=until - timedelta(days=27)
    url=f"{META_BASE}/{acct}/insights"
    params={
        "level":"account",
        "breakdowns":"hourly_stats_aggregated_by_advertiser_time_zone",
        "time_range": json.dumps({"since": since.isoformat(), "until": until.isoformat()}),
        "fields":"impressions,clicks,actions",
        "access_token": tok,
    }
    r=requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    js=r.json()
    # Build per-hour clicks/purchases
    hours={str(h).zfill(2): {"clicks":0.0, "purch":0.0} for h in range(24)}
    for row in js.get("data", []):
        # rows include field hourly_stats_aggregated_by_advertiser_time_zone like "00", "01", ...
        raw=row.get("hourly_stats_aggregated_by_advertiser_time_zone")
        # Values may be "00" or "00:00:00 - 00:59:59"; normalize to HH
        if not raw:
            continue
        h = raw[:2]
        if not h: continue
        clicks=float(row.get("clicks") or 0.0)
        purch=0.0
        for a in (row.get("actions") or []):
            if a.get("action_type")=="offsite_conversion.fb_pixel_purchase": purch=float(a.get("value") or 0.0)
        hours[h]["clicks"]+=clicks; hours[h]["purch"]+=purch
    # Compute rates and normalize to mean 1.0
    rates={h:( (v["purch"]/v["clicks"]) if v["clicks"]>0 else 0.0) for h,v in hours.items()}
    base=np.mean([r for r in rates.values() if r>0]) if any(r>0 for r in rates.values()) else 0.0
    mults={h:( (rate/base) if base>0 else 1.0) for h,rate in rates.items()}
    # Smooth extremes
    for h in mults:
        mults[h]=float(np.clip(mults[h], 0.7, 1.3))
    out={"since": since.isoformat(), "until": until.isoformat(), "hourly": mults}
    os.makedirs("AELP2/reports", exist_ok=True)
    path="AELP2/reports/hourly_multipliers.json"
    open(path,"w").write(json.dumps(out,indent=2))
    print(path)

if __name__ == "__main__":
    main()
