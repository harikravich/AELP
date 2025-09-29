#!/usr/bin/env python3
from __future__ import annotations
import os, json
from pathlib import Path
import numpy as np
from google.cloud import bigquery

ROOT = Path(__file__).resolve().parents[2]
ED = ROOT / 'AELP2' / 'reports' / 'creative_visual_embeddings'

def main():
    project = os.getenv('GOOGLE_CLOUD_PROJECT') or 'aura-thrive-platform'
    dataset = os.getenv('BIGQUERY_TRAINING_DATASET') or 'gaelp_training'
    table = f"{project}.{dataset}.creative_visual_embeddings"
    bq = bigquery.Client(project=project)
    idx = json.loads((ED / 'index.json').read_text()) if (ED / 'index.json').exists() else {}
    rows=[]
    for stem, path in idx.items():
        p=Path(path)
        if not p.exists():
            continue
        emb=np.load(p)
        rows.append({
            'stem': stem,
            'model': 'openclip/ViT-B-32',
            'dim': int(emb.shape[0]),
            'embedding': emb.tobytes()
        })
    if not rows:
        print(json.dumps({'status':'no_rows'})); return
    errors=bq.insert_rows_json(table, rows)
    if errors:
        print(json.dumps({'status':'bq_errors','errors':errors})); return
    print(json.dumps({'status':'ok','inserted':len(rows)}))

if __name__=='__main__':
    main()

