#!/usr/bin/env bash
set -euo pipefail

# One-command pipeline: Creative-aware CTR -> Priors -> Contextual TS -> Slate
# Safe to run on a laptop; heavy deps are optional and features degrade gracefully.

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

UNIFIED=${UNIFIED:-artifacts/synth/unified.parquet}
ASSETS_DIR=${ASSETS_DIR:-assets/demo_creatives}
CRE_FEATS=${CRE_FEATS:-artifacts/creative/creative_features.parquet}
JOINED=${JOINED:-artifacts/features/ctr_joined.parquet}
CTR_MODEL=${CTR_MODEL:-artifacts/models/ctr_creative.joblib}
CTR_PREDS=${CTR_PREDS:-artifacts/predictions/ctr_scores.parquet}
NOVEL_JOINED=${NOVEL_JOINED:-artifacts/features/novel_catalog.parquet}
NOVEL_PREDS=${NOVEL_PREDS:-artifacts/predictions/novel_ctr_scores.parquet}
PRIORS=${PRIORS:-artifacts/priors/priors.json}
TS_OUT=${TS_OUT:-artifacts/priors/ts_strategies_ctr.json}
TS_NOVEL_OUT=${TS_NOVEL_OUT:-artifacts/priors/ts_strategies_novel.json}
SLATE_OUT=${SLATE_OUT:-artifacts/slates/creative_slate.json}

echo "[1/8] Generate demo creatives + manifest (if needed)"
if [[ ! -f "$ASSETS_DIR/ads_manifest.csv" ]]; then
  python3 pipelines/creative/generate_demo_assets.py --out-dir "$ASSETS_DIR" --count 12
fi

echo "[2/8] Extract features for manifest"
PYTHONPATH="$ROOT" python3 pipelines/creative/extract_for_manifest.py \
  --manifest "$ASSETS_DIR/ads_manifest.csv" \
  --out "$CRE_FEATS"

echo "[3/8] Build/join features"
PYTHONPATH="$ROOT" python3 pipelines/features/join_creative.py \
  --unified "$UNIFIED" \
  --creative "$CRE_FEATS" \
  --out "$JOINED"

echo "[4/8] Train creative-aware CTR model"
PYTHONPATH="$ROOT" python3 pipelines/ctr/train_ctr_creative.py \
  --data "$JOINED" \
  --out "$CTR_MODEL"

echo "[5/8] Predict CTR for joined table"
PYTHONPATH="$ROOT" python3 pipelines/ctr/predict_ctr.py \
  --model "$CTR_MODEL" \
  --data "$JOINED" \
  --out "$CTR_PREDS"

echo "[5b] Predict CTR for novel creatives (no history)"
PYTHONPATH="$ROOT" python3 pipelines/features/catalog_from_creatives.py \
  --creative "$CRE_FEATS" \
  --out "$NOVEL_JOINED"
PYTHONPATH="$ROOT" python3 pipelines/ctr/predict_ctr.py \
  --model "$CTR_MODEL" \
  --data "$NOVEL_JOINED" \
  --out "$NOVEL_PREDS"

echo "[6/8] Compose Beta priors (CTR)"
PYTHONPATH="$ROOT" python3 pipelines/priors/compose_beta_priors.py \
  --data "$UNIFIED" \
  --groupby ad_id \
  --out "$PRIORS"

echo "[7/8] Build contextual TS strategies"
PYTHONPATH="$ROOT" python3 pipelines/bandit/contextual_ts.py \
  --features "$JOINED" \
  --preds "$CTR_PREDS" \
  --priors "$PRIORS" \
  --out "$TS_OUT"
PYTHONPATH="$ROOT" python3 pipelines/bandit/contextual_ts.py \
  --features "$NOVEL_JOINED" \
  --preds "$NOVEL_PREDS" \
  --priors "$PRIORS" \
  --out "$TS_NOVEL_OUT"

echo "[8/8] Build slate from creatives (uses combined score + CLIP diversity if present)"
PYTHONPATH="$ROOT" python3 pipelines/slate/build_slate.py \
  --assets "$ASSETS_DIR" \
  --features "$CRE_FEATS" \
  --k 6 \
  --out "$SLATE_OUT"

echo "Done. Artifacts:"
echo "  - Creative features: $CRE_FEATS"
echo "  - Joined features:   $JOINED"
echo "  - CTR model:         $CTR_MODEL"
echo "  - CTR predictions:   $CTR_PREDS"
echo "  - Novel predictions: $NOVEL_PREDS"
echo "  - Beta priors:       $PRIORS"
echo "  - TS strategies:     $TS_OUT"
echo "  - TS (novel):        $TS_NOVEL_OUT"
echo "  - Slate:             $SLATE_OUT"
