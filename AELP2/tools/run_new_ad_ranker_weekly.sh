#!/usr/bin/env bash
set -euo pipefail
# Weekly retrain + refresh pipeline for Meta-only new-ad scorer
python3 AELP2/tools/build_features_from_creative_objects.py
python3 AELP2/tools/build_labels_weekly.py
python3 AELP2/tools/train_new_ad_ranker.py
python3 AELP2/tools/eval_wbua_forward.py
python3 AELP2/tools/eval_wbua_novel.py
python3 AELP2/tools/eval_wbua_cluster.py
python3 AELP2/tools/score_new_ads.py || true
python3 AELP2/tools/generate_score_loop.py || true
echo "done"

