# Meta-only New Ad Scorer + Holdouts (Runbook)

What this does
- Trains a lightweight pairwise ranker on weekly Aura data to predict whether a variant will beat a historical baseline (WBUA task).
- Provides forward, novel, and cluster holdout reports.
- Scores local candidate videos (e.g., MJ/Runway finals) and emits a Top-K slate with conformal lower bounds.

Key commands
- Build features from cached Meta creative objects:
  - `python3 AELP2/tools/build_features_from_creative_objects.py`
- Build training pairs (diff vs baseline):
  - `python3 AELP2/tools/build_labels_weekly.py`
- Train + calibrate + conformal:
  - `python3 AELP2/tools/train_new_ad_ranker.py`
- Holdouts (accuracy):
  - `python3 AELP2/tools/eval_wbua_forward.py`
  - `python3 AELP2/tools/eval_wbua_novel.py`
  - `python3 AELP2/tools/eval_wbua_cluster.py`
- Score current finals and select Top-K:
  - `python3 AELP2/tools/score_new_ads.py`
  - `python3 AELP2/tools/generate_score_loop.py`

Artifacts
- Models: `AELP2/models/new_ad_ranker/`
- Reports:
  - `AELP2/reports/wbua_forward.json`
  - `AELP2/reports/wbua_novel_summary.json`
  - `AELP2/reports/wbua_cluster_summary.json`
  - `AELP2/reports/new_ad_scores.json`
  - `AELP2/reports/topk_slate.json`

Notes
- CLIP embeddings are optional; the current extractor falls back to color-histogram embeddings if OpenCLIP/TorchVision are not available in this environment.
- Object/hand detection via YOLOv8 is installed but not used by the v0 ranker; wire it into features if desired.
- For real Meta Ad Library crawling, set up read-only tokens or run `AELP2/tools/fetch_meta_adlibrary.py` with proper scopes; the current script uses cached repo objects for schema only.

