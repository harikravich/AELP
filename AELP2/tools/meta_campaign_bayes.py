#!/usr/bin/env python3
"""
Bayesian per-campaign MMM (Poisson with geometric adstock) for Meta campaigns.

Inputs:
- META_ACCESS_TOKEN, META_ACCOUNT_ID in environment or .env (export VAR=... lines)

Outputs (stdout JSON):
{
  "created_at": "...",
  "campaigns": [
     {
       "id": "...",
       "name": "...",
       "metrics_14d": {...},
       "current_spend_med7": float,
       "posterior": {
          "marginal_cac": {"p10":..., "p50":..., "p90":...},
          "s_for_cac": {
             "100": {"p10":..., "p50":..., "p90":...},
             "120": {...},
             "150": {...}
          }
       }
     }, ...
  ]
}

Note: This script does not modify any live campaigns; it only reads insights.
"""
from __future__ import annotations

import os
import sys
import json
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import requests
import numpy as np

# Use the venv's jax/numpyro if available
try:
    import jax
    import jax.numpy as jnp
    from jax import lax
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS
except Exception as e:
    print(json.dumps({"error": f"Missing JAX/NumPyro: {e}"}))
    sys.exit(1)


def load_env_from_dotenv(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line=line[len("export "):]
                if "=" not in line:
                    continue
                k,v=line.split("=",1)
                os.environ.setdefault(k.strip(), v.strip())
    except Exception:
        pass


def get_actions_value(row: Dict, key: str) -> float:
    for a in (row.get('actions') or []):
        if a.get('action_type')==key:
            try:
                return float(a.get('value') or 0.0)
            except Exception:
                return 0.0
    return 0.0


def fetch_campaigns_last14(token: str, acct: str) -> List[Dict]:
    base=f"https://graph.facebook.com/v21.0/{acct}/insights"
    params={
        'date_preset':'last_14d','level':'campaign',
        'fields':'campaign_id,campaign_name,spend,impressions,clicks,actions,frequency',
        'access_token': token
    }
    r=requests.get(base, params=params, timeout=90).json()
    out=[]
    for d in r.get('data',[]):
        purch=get_actions_value(d,'offsite_conversion.fb_pixel_purchase')
        spend=float(d.get('spend') or 0.0)
        clicks=int(float(d.get('clicks') or 0.0)) if d.get('clicks') else 0
        impr=int(float(d.get('impressions') or 0.0)) if d.get('impressions') else 0
        ctr=(clicks/impr*100) if impr else None
        cpc=(spend/clicks) if clicks else None
        freq=float(d.get('frequency') or 0.0)
        out.append({'id':d['campaign_id'],'name':d['campaign_name'],'purch_14d':purch,'spend_14d':spend,'ctr_14d':ctr,'cpc_14d':cpc,'freq_7d':freq})
    return out


def fetch_daily_series(token: str, acct: str, campaign_id: str, days: int=90) -> Tuple[str, List[Tuple[str,float,float]]]:
    base=f"https://graph.facebook.com/v21.0/{acct}/insights"
    params={'time_increment':1,'date_preset':'last_90d','level':'campaign','fields':'campaign_id,campaign_name,date_start,spend,actions','access_token':token}
    url=base; out=[]; name=None
    for _ in range(40):
        r=requests.get(url, params=params if url==base else None, timeout=120).json()
        for row in r.get('data',[]):
            if row.get('campaign_id')!=campaign_id:
                continue
            name=row.get('campaign_name', name)
            spend=float(row.get('spend') or 0.0)
            purch=get_actions_value(row,'offsite_conversion.fb_pixel_purchase')
            out.append((row.get('date_start'), spend, purch))
        next_url=(r.get('paging') or {}).get('next')
        if not next_url:
            break
        url=next_url; params=None
    out.sort(key=lambda x:x[0])
    if len(out)>days:
        out=out[-days:]
    return name or campaign_id, out


def adstock_scan(spend: jnp.ndarray, decay: float) -> jnp.ndarray:
    def step(carry, s):
        new = s + decay * carry
        return new, new
    last, seq = lax.scan(step, 0.0, spend)
    return seq


def mmm_model(spend: jnp.ndarray, conv: jnp.ndarray):
    # Priors
    alpha = numpyro.sample('alpha', dist.Normal(0.0, 5.0))
    beta = numpyro.sample('beta', dist.HalfNormal(2.0))  # concave when <1 typically learned
    # Bound decay to < 0.95 for stability
    u = numpyro.sample('u_decay', dist.Beta(2.0, 2.0))
    decay = 0.95 * u
    eps = numpyro.sample('eps', dist.Exponential(10.0))  # small offset

    xs = adstock_scan(spend, decay)
    rate = jnp.exp(alpha + beta * jnp.log(xs + eps))
    numpyro.sample('y', dist.Poisson(rate), obs=conv)


def run_mcmc(spend: np.ndarray, conv: np.ndarray, seed: int=0, warmup: int=600, samples: int=1000):
    kernel = NUTS(mmm_model, target_accept_prob=0.85)
    mcmc = MCMC(kernel, num_warmup=warmup, num_samples=samples, num_chains=2, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), jnp.asarray(spend), jnp.asarray(conv))
    return mcmc.get_samples()


def summarize_headroom(samples: Dict[str,jnp.ndarray], spend_series: np.ndarray, grid_mult=(0.5, 2.0)) -> Dict:
    # Current spend ~ median of last 7 days
    cur = float(np.median(spend_series[-7:])) if len(spend_series)>=7 else float(np.mean(spend_series))
    # Carry from last day under each decay sample
    s = jnp.asarray(spend_series)
    alpha = samples['alpha']; beta = samples['beta']; u = samples['u_decay']; eps = samples['eps']
    decay = 0.95 * u

    def carry_last(dec):
        xs = adstock_scan(s, dec)
        return xs[-1]

    carrys = jax.vmap(carry_last)(decay)

    # Spend grid around current
    gmin = max(1.0, 0.5 * cur)
    gmax = max(gmin+1.0, 2.0 * cur)
    grid = np.linspace(gmin, gmax, 25)

    # Predict conversions for next day given proposed spend and last carry
    def pred_for(s_one):
        xs_next = s_one + decay * carrys
        lam = jnp.exp(alpha + beta * jnp.log(xs_next + eps))
        return lam

    lam_grid = jnp.stack([pred_for(float(z)) for z in grid], axis=1)  # [S, G]

    # CAC distribution across grid
    # avoid divide by zero
    cac_grid = jnp.asarray(grid)[None, :] / jnp.clip(lam_grid, a_min=1e-6)

    # Marginal CAC near current via finite difference
    delta = max(1.0, 0.05 * cur)
    lam_cur = pred_for(cur)
    lam_up = pred_for(cur + delta)
    marg_cac = delta / jnp.clip((lam_up - lam_cur), a_min=1e-6)

    def pct(v, q):
        return float(np.percentile(np.asarray(v), q))

    # helpers to find spend for CAC thresholds using median CAC curve
    median_cac = np.median(np.asarray(cac_grid), axis=0)
    def solve_cac(threshold):
        for s_val, c_val in zip(grid, median_cac):
            if c_val <= threshold:
                return float(s_val)
        return None

    # CI for marginal CAC
    out = {
        'current_spend_med7': cur,
        'marginal_cac': {
            'p10': pct(marg_cac,10), 'p50': pct(marg_cac,50), 'p90': pct(marg_cac,90)
        },
        'grid': grid.tolist(),
        'cac_curve': {
            'p10': np.percentile(np.asarray(cac_grid),10,axis=0).tolist(),
            'p50': np.percentile(np.asarray(cac_grid),50,axis=0).tolist(),
            'p90': np.percentile(np.asarray(cac_grid),90,axis=0).tolist(),
        },
        's_for_cac': {
            '150': solve_cac(150.0),
            '120': solve_cac(120.0),
            '100': solve_cac(100.0),
        }
    }
    return out


def main():
    load_env_from_dotenv()
    token=os.getenv('META_ACCESS_TOKEN')
    acct=os.getenv('META_ACCOUNT_ID')
    if not token or not acct:
        print(json.dumps({'error':'Missing META_ACCESS_TOKEN or META_ACCOUNT_ID'}))
        sys.exit(1)

    camps=fetch_campaigns_last14(token, acct)
    # Prioritize by purchases; include up to 8
    camps_sorted=sorted(camps, key=lambda x:x.get('purch_14d',0.0), reverse=True)[:8]

    results=[]
    for c in camps_sorted:
        name, daily = fetch_daily_series(token, acct, c['id'], days=90)
        spend=np.array([s for _,s,_ in daily], dtype=float)
        conv=np.array([p for _,_,p in daily], dtype=float)
        # Skip if little data
        if (spend>0).sum()<20 or conv.sum()<20:
            results.append({
                'id': c['id'], 'name': name, 'metrics_14d': c,
                'note': 'Insufficient recent data for robust Bayesian fit'
            })
            continue
        try:
            samples=run_mcmc(spend, conv, seed=0, warmup=500, samples=800)
        except Exception as e:
            results.append({
                'id': c['id'], 'name': name, 'metrics_14d': c,
                'error': f'MCMC failed: {e}'
            })
            continue
        head=summarize_headroom(samples, spend)
        results.append({
            'id': c['id'], 'name': name,
            'metrics_14d': c,
            'headroom': head
        })

    print(json.dumps({'created_at': time.strftime('%Y-%m-%d %H:%M:%S %Z'), 'campaigns': results}, indent=2))


if __name__=='__main__':
    main()

