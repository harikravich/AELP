#!/usr/bin/env python3
"""
PMax Bandit Service (shadow): logs decisions for Performance Max campaigns.
Uses ads_campaign_performance filtered by advertising_channel_sub_type='PERFORMANCE_MAX'.
Falls back to dry-run if table/field missing.
"""
import os, json, argparse
from datetime import datetime
from typing import List, Dict, Any
import numpy as np
from google.cloud import bigquery


def fetch_pmax(bq: bigquery.Client, project: str, dataset: str) -> List[Dict[str,Any]]:
    sql = f"""
      SELECT CAST(campaign_id AS STRING) AS id, SUM(impressions) AS imps, SUM(clicks) AS clicks
      FROM `{project}.{dataset}.ads_campaign_performance`
      WHERE advertising_channel_sub_type = 'PERFORMANCE_MAX'
        AND DATE(date) BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY) AND CURRENT_DATE()
      GROUP BY id
      HAVING imps > 0
      ORDER BY imps DESC LIMIT 20
    """
    try:
        return [dict(r) for r in bq.query(sql).result()]
    except Exception:
        return []


def select(arms: List[Dict[str,Any]]):
    best=None; bests=-1.0; ann=[]
    for a in arms:
        imps = float(a.get('imps',0.0)); clk=float(a.get('clicks',0.0))
        alpha=1+clk; beta=1+max(0.0,imps-clk)
        s=float(np.random.beta(alpha,beta)); row={**a,'prior_alpha':alpha,'prior_beta':beta,'posterior_alpha':alpha,'posterior_beta':beta,'sample':s}
        ann.append(row)
        if s>bests: best=row; bests=s
    return best or {}, ann


def main():
    p=argparse.ArgumentParser(); p.add_argument('--dry_run',action='store_true'); args=p.parse_args()
    project=os.getenv('GOOGLE_CLOUD_PROJECT'); dataset=os.getenv('BIGQUERY_TRAINING_DATASET'); now=datetime.utcnow().isoformat()
    if args.dry_run or not (project and dataset):
        arms=[{'id':'pmax1','imps':10000,'clicks':300},{'id':'pmax2','imps':8000,'clicks':200}]
        sel,_=select(arms); print(json.dumps({'selected': sel.get('id'),'sample':sel.get('sample',0.0)})); return
    bq=bigquery.Client(project=project)
    arms=fetch_pmax(bq,project,dataset)
    if not arms:
        print('no pmax campaigns'); return
    sel, ann = select(arms)
    row={
        'timestamp':now,'platform':'google_ads','channel':'pmax','campaign_id':sel.get('id'),'ad_id':sel.get('id'),
        'prior_alpha':sel.get('prior_alpha'),'prior_beta':sel.get('prior_beta'),'posterior_alpha':sel.get('posterior_alpha'),
        'posterior_beta':sel.get('posterior_beta'),'sample':sel.get('sample'),'context':json.dumps({'arms':ann[:5]})
    }
    bq.insert_rows_json(f"{project}.{dataset}.bandit_decisions",[row])
    print('pmax bandit logged')

if __name__=='__main__':
    main()

