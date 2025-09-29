#!/usr/bin/env python3
from __future__ import annotations
import os, json, sys
from pathlib import Path
from google.cloud import bigquery

ROOT = Path(__file__).resolve().parents[2]
RD = ROOT / 'AELP2' / 'reports' / 'creative_enriched'

def main():
    project = os.getenv('GOOGLE_CLOUD_PROJECT') or 'aura-thrive-platform'
    dataset = os.getenv('BIGQUERY_TRAINING_DATASET') or 'gaelp_training'
    table = f"{project}.{dataset}.creative_visual_features"
    path = RD / 'finals_features.jsonl'
    if not path.exists():
        print(json.dumps({'status':'missing','path':str(path)})); return
    bq = bigquery.Client(project=project)
    rows=[]
    with path.open() as f:
        for ln in f:
            d=json.loads(ln)
            rows.append({
                'file': d.get('file'),
                'motion_mean': d.get('motion_mean'),
                'motion_std': d.get('motion_std'),
                'lap_var': d.get('lap_var'),
                'tenengrad': d.get('tenengrad'),
                'colorfulness': d.get('colorfulness'),
                'legibility': d.get('legibility'),
                'persons': d.get('persons'),
                'phones': d.get('phones'),
                'lufs': d.get('lufs'),
            })
    errors = bq.insert_rows_json(table, rows)
    if errors:
        print(json.dumps({'status':'bq_errors','errors':errors})); sys.exit(2)
    print(json.dumps({'status':'ok','inserted':len(rows)}))

if __name__=='__main__':
    main()

