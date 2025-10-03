#!/usr/bin/env bash
set -euo pipefail

# GPU Creative-aware CTR pipeline (CLIP + YOLO if CUDA available)
# Usage:
#   export META_ACCESS_TOKEN=... META_ACCOUNT_ID=act_...
#   bash scripts/gpu_creative_ctr.sh

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

if [[ -f .env ]]; then
  # shellcheck disable=SC1091
  source .env || true
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[warn] nvidia-smi not found; running in CPU mode."
fi

if [[ "${INSTALL_GPU_DEPS:-0}" == "1" ]]; then
  echo "[deps] Installing GPU requirements..."
  pip install -r requirements/requirements-gpu.txt
fi

UNIFIED=${UNIFIED:-artifacts/marketing/unified_ctr.parquet}
MANIFEST=${MANIFEST:-assets/meta_creatives/ads_manifest.csv}
CRE_FEATS=${CRE_FEATS:-artifacts/creative/meta_creative_features.parquet}
JOINED=${JOINED:-artifacts/features/marketing_ctr_joined.parquet}
CTR_MODEL=${CTR_MODEL:-artifacts/models/ctr_creative_marketing.joblib}
CTR_PREDS=${CTR_PREDS:-artifacts/predictions/ctr_scores.parquet}
PRIORS=${PRIORS:-artifacts/priors/priors.json}
TS_OUT=${TS_OUT:-artifacts/priors/ts_strategies_ctr.json}
SLATE_OUT=${SLATE_OUT:-artifacts/slates/ad_slate.json}

echo "[1/6] Extract creative features (GPU if available)"
ENABLE_CLIP=1 PYTHONPATH="$ROOT" python3 pipelines/creative/extract_for_manifest.py \
  --manifest "$MANIFEST" \
  --out "$CRE_FEATS"

echo "[2/6] Join creative features with unified metrics"
PYTHONPATH="$ROOT" python3 pipelines/features/join_creative.py \
  --unified "$UNIFIED" \
  --creative "$CRE_FEATS" \
  --out "$JOINED"

echo "[3/6] Train creative-aware CTR model"
PYTHONPATH="$ROOT" python3 pipelines/ctr/train_ctr_creative.py \
  --data "$JOINED" \
  --out "$CTR_MODEL"

echo "[4/6] Forward holdout eval"
PYTHONPATH="$ROOT" python3 pipelines/validation/ctr_forward_eval.py \
  --data "$JOINED" \
  --train-end "${TRAIN_END:-2025-09-26}" \
  --holdout-start "${HOLDOUT_START:-2025-09-27}" \
  --out artifacts/validation/ctr_forward_marketing.json || true

echo "[5/6] Predict + build contextual TS"
python3 - << 'PY'
import pandas as pd
from pathlib import Path
j='artifacts/features/marketing_ctr_joined.parquet'
df=pd.read_parquet(j)
df2=df.sort_values(['ad_id','date']).drop_duplicates('ad_id', keep='last')
Path('artifacts/features/marketing_ctr_joined_latest.parquet').parent.mkdir(parents=True, exist_ok=True)
df2.to_parquet('artifacts/features/marketing_ctr_joined_latest.parquet', index=False)
print('latest rows',len(df2))
PY
PYTHONPATH="$ROOT" python3 pipelines/ctr/predict_ctr.py \
  --model "$CTR_MODEL" \
  --data artifacts/features/marketing_ctr_joined_latest.parquet \
  --out "$CTR_PREDS"
PYTHONPATH="$ROOT" python3 pipelines/bandit/contextual_ts.py \
  --features artifacts/features/marketing_ctr_joined_latest.parquet \
  --preds "$CTR_PREDS" \
  --priors "$PRIORS" \
  --out "$TS_OUT"

echo "[6/6] Build slate"
PYTHONPATH="$ROOT" python3 pipelines/slate/build_ad_slate.py \
  --ts "$TS_OUT" \
  --manifest "$MANIFEST" \
  --k 30 \
  --out "$SLATE_OUT"

echo "Done. Artifacts:"
echo "  - Features: $CRE_FEATS"
echo "  - Joined:   $JOINED"
echo "  - Model:    $CTR_MODEL"
echo "  - Preds:    $CTR_PREDS"
echo "  - TS:       $TS_OUT"
echo "  - Slate:    $SLATE_OUT"

