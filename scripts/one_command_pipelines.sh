#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
ART="$ROOT/artifacts"
mkdir -p "$ART/demo_data" "$ART/models" "$ART/slates" "$ART/landers" "$ART/priors" "$ART/validation" "$ART/generation"

echo "[1/9] Generate tiny synthetic Meta + GA"
python3 - <<'PY'
import pandas as pd, numpy as np, os
art=os.environ['ART']
meta=pd.DataFrame([
    {"date":"2025-09-01","campaign_id":"c1","adset_id":"s1","ad_id":"a1","device":"mobile","impressions":1000,"clicks":50,"spend":25.0},
    {"date":"2025-09-02","campaign_id":"c1","adset_id":"s1","ad_id":"a2","device":"desktop","impressions":800,"clicks":20,"spend":12.0},
])
ga=pd.DataFrame([
    {"date":"2025-09-01","campaign_id":"c1","adset_id":"s1","ad_id":"a1","device":"mobile","sessions":80,"transactions":3,"subscriptions":4},
    {"date":"2025-09-02","campaign_id":"c1","adset_id":"s1","ad_id":"a2","device":"desktop","sessions":60,"transactions":1,"subscriptions":1},
])
meta.to_csv(f"{art}/demo_data/meta.csv",index=False)
ga.to_csv(f"{art}/demo_data/ga.csv",index=False)
print("wrote demo CSVs")
PY

echo "[2/9] Unify datasets"
python3 "$ROOT/pipelines/data/unify_meta_ga.py" --meta "$ART/demo_data/meta.csv" --ga "$ART/demo_data/ga.csv" --out "$ART/unified_training.parquet"

echo "[3/9] Train CTR baseline + fine-tune"
python3 "$ROOT/pipelines/ctr/train_merlin_criteo.py" --data "$ART/demo_data/meta.csv" --out "$ART/models/merlin_criteo_ctr.joblib"
python3 "$ROOT/pipelines/ctr/finetune_aura_ctr.py" --base "$ART/models/merlin_criteo_ctr.joblib" --data "$ART/demo_data/meta.csv" --out "$ART/models/aura_ctr_calibrated.joblib"

echo "[4/9] Train GA heads"
python3 "$ROOT/pipelines/ga/train_heads.py" --data "$ART/unified_training.parquet" --outdir "$ART/models/ga_heads"

echo "[5/9] Priors -> Thompson"
python3 "$ROOT/pipelines/priors/compose_beta_priors.py" --data "$ART/unified_training.parquet" --out "$ART/priors/priors.json"
python3 "$ROOT/pipelines/priors/integrate_priors_thompson.py" --priors "$ART/priors/priors.json" --out "$ART/priors/ts_strategies.json"

echo "[6/9] Generate landers + extract features"
python3 "$ROOT/pipelines/landers/generate_landers.py" --config "$ROOT/pipelines/landers/sample_landers.yaml" --outdir "$ART/landers"
python3 "$ROOT/pipelines/creative/extract_features.py" --in "$ART/landers" --out "$ART/generation/creative_features.parquet"

echo "[7/9] Build slate"
python3 "$ROOT/pipelines/slate/build_slate.py" --assets "$ART/landers" --features "$ART/generation/creative_features.parquet" --k 2 --out "$ART/slates/lander_slate.json"

echo "[8/9] Validation + uplift demo"
python3 "$ROOT/pipelines/validation/forward_holdout.py" --data "$ART/unified_training.parquet" --out "$ART/validation/forward_holdout.json"
python3 - <<'PY'
import pandas as pd, os, json
art=os.environ['ART']
live=pd.DataFrame([{'asset':'a','impressions':1000,'conversions':20},{'asset':'b','impressions':1000,'conversions':25}])
new=pd.DataFrame([{'asset':'a','impressions':1000,'conversions':24},{'asset':'b','impressions':1000,'conversions':30}])
live.to_csv(f"{art}/validation/live.csv",index=False)
new.to_csv(f"{art}/validation/new.csv",index=False)
PY
python3 "$ROOT/pipelines/eval/head_to_head.py" --live "$ART/validation/live.csv" --new "$ART/validation/new.csv" --out "$ART/validation/uplift_report.json"

echo "[9/9] Publish locally (public/)"
python3 "$ROOT/tools/publish_gcs.py" --src "$ART/landers" --dest gs://demo/aura/landers/v1

echo "Done. See artifacts/ and public/ for outputs."
