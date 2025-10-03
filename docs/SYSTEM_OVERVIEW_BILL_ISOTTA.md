# AELP/GAELP System Overview (Bill + Isotta)

This document gives you a complete map of environments, data, models, and pipelines so you can take over: (1) evaluating current Meta ads, (2) increasing volume while lowering CAC, and (3) simulating new ad concepts to forecast probability of success.

Updated: 2025-10-02

## 1) Environments & Access

### Servers (Compute Engine)
- merlin-l4-1 (us-central1-c): GPU L4; training/scoring, feature extraction; shared env/venv configured.
- thrive-backend (us-central1-a): Backend host with nginx + containers; staging/prod API endpoints.
- aelp-sim-rl-1 (us-central1-a): Simulation/RL CPU box for batch jobs.

### GPU VM (primary)
- Hostname: `merlin-l4-1`
- Zone/Region: `us-central1-c` / `us-central1`
- Purpose: Model training/scoring, feature extraction (CLIP+YOLO), daily evaluation runs
- SSH:
  - Bill: `gcloud compute ssh bill@merlin-l4-1 --zone us-central1-c`
  - Isotta: `gcloud compute ssh isotta@merlin-l4-1 --zone us-central1-c`
- Code path: `/home/harikravich_gmail_com/AELP`
- Shared venv: `/srv/aelp/.venv` (activate with `source /srv/aelp/.venv/bin/activate`)
- Env vars (all keys pre-loaded): `/srv/aelp/.env.local` (auto-sourced for `aelp` group)
- Group access: you’re both in the `aelp` group; repo is group-readable; `artifacts/` is group-writable

### Dashboard (AELP2, Cloud Run)
- Repo path: `AELP2/apps/dashboard`
- Deploy target: Cloud Run (us-central1)
- Env example: `AELP2/apps/dashboard/.env.example` (NextAuth, allowed domain)
- Purpose: internal status, reports, run controls (optional for your first week)

### GCP Services (project: aura-thrive-platform)
- BigQuery datasets (canonical names used throughout docs):
  - `gaelp_training` (primary training/analytics)
  - `gaelp_users` (user-level or cohort views)
  - Optional sandbox/stage/prod variants per `AELP2/docs/HANDOFF_README.md`
- Storage (GCS) buckets:
  - Discovered in project:
    - `gs://gaelp-model-checkpoints-hariravichandran` (primary)
      - prefixes: `coldstart/`, `checkpoints/`, `creative/`, `creatives/`, `balance-assets/`, `balance-landers/`
      - latest example: `coldstart/20251001_0546/outputs/finals/*.mp4/*.json/*.html`
    - `gs://gaelp-model-checkpoints-1755753776` (present, currently empty/unused)
    - `gs://aelp-repo-drop-556751870393` (project scratch)
    - `gs://run-sources-aura-thrive-platform-us-central1` (Cloud Run build sources)
  - Planned per infra docs: `aelp2-{dev,stage,prod}-{reports,artifacts}`
  - Tools: `tools/publish_gcs.py`, `scripts/checks/list_gcs_inventory.sh`
- Redis (optional): two instances referenced in infra docs for caching/session

### Cloud Run Services (us-central1)
- aelp2-dashboard → internal dashboard (Next.js)
- thrive-simple-backend → lightweight backend API
- thrive-ui-universal → Thrive UI renderer
- thrive-mcp-service → MCP connectors service
See also: `docs/SERVERS_AND_ENDPOINTS.md` for URLs.

## 2) Code Layout (what’s in git)

- Root repo (`/home/harikravichandran/AELP`)
  - `pipelines/` — end-to-end ML and data jobs (see section 4)
  - `tools/meta/` — Meta Ads utilities (e.g., `fetch_ads_and_insights.py`)
  - `artifacts/` — local model outputs, predictions, and features
  - `assets/` — local creative assets for scoring (e.g., videos)
  - `training_orchestrator/` — multi-phase sim→real progression (for later)
  - `docs/` — this doc plus onboarding
  - `scripts/` — dev/admin/secrets checks and helpers

- AELP2 (productized components)
  - `AELP2/apps/dashboard` — internal dashboard (Next.js/Cloud Run)
  - `AELP2/requirements_dev_310.txt` — pinned Python stack for 3.10
  - `AELP2/docs/*` — Handoff, security, onboarding guides
  - Infra: `infrastructure/` (Terraform, GKE/VPC, BigQuery schemas and notes)

## 3) Data & Assets

### Local (GPU VM)
- Features (video & creative):
  - `artifacts/creative/*.parquet` (CLIP+YOLO+temporal features)
- Predictions:
  - `artifacts/predictions/ctr_scores_reg.parquet` (CTR regressor outputs)
  - `artifacts/predictions/ctr_scores.parquet` (classifier p(click>0) if used)
  - `artifacts/predictions/current_running_scored.parquet` (daily scored table)
  - `artifacts/predictions/current_running_scored_unique.parquet` (unique link CTR version)
  - For new Balance videos/thumbs: `artifacts/predictions/veo_videos_ctr.parquet`, `.../veo_balance_ctr.parquet`
- Unified daily marketing table (local parquet):
  - `artifacts/marketing/unified_ctr.parquet`
  - Columns (link-only KPI): `impressions`, `inline_link_clicks`, `unique_inline_link_clicks`, `ctr`, `link_ctr`, `link_ctr_unique`, plus spend-derived fields

### Remote (GCS)
- Model checkpoints and generated videos (examples):
  - `gs://gaelp-model-checkpoints-hariravichandran/coldstart/...`
- Project buckets (reports/artifacts) as in AELP2 handoff docs

### BigQuery
- Datasets: `gaelp_training`, `gaelp_users` (per AELP2 docs)
- Typical tables (if/when loaded):
  - `campaigns`, `performance_metrics`, `agent_episodes`, `simulation_data`, `personas`, `safety_events`
  - Views: `campaign_performance_summary`, `agent_performance_comparison`, etc. (see `infrastructure/bigquery/README.md`)

## 4) Pipelines (what they do)

Creative/CTR stack (Meta)
- Fetch Meta ads+insights (read-only):
  - `tools/meta/fetch_ads_and_insights.py`
  - Output: `artifacts/meta/{ads.csv, insights.csv}` with link and unique_link fields
- Unify daily metrics (per-ad, per-day):
  - `pipelines/data/unify_meta_marketing.py` → `artifacts/marketing/unified_ctr.parquet`
  - Computes `ctr`, `link_ctr`, `link_ctr_unique` (our scrubbed KPI)
- Join creative features (CLIP/YOLO/etc.):
  - `pipelines/features/join_creative.py` + `pipelines/features/enhance_features.py`
  - Output: `artifacts/features/marketing_ctr_enhanced.parquet` (+ latest variants)
- Train models:
  - Regressor (CTR_u): `pipelines/ctr/train_ctr_creative.py` → `artifacts/models/ctr_creative_enhanced.joblib`
  - Classifier (1{unique_link_click>0}): `pipelines/ctr/train_ctr_classifier.py` → `artifacts/models/ctr_classifier.joblib`
  - Eval: `pipelines/validation/{ctr_forward_eval.py, forward_holdout_classifier.py, ranking_eval.py}`
- Predict & score current running ads:
  - `pipelines/ctr/predict_ctr.py` → `artifacts/predictions/ctr_scores_reg.parquet`
  - “Current running” join/report written to `artifacts/predictions/current_running_scored_unique.parquet`
- Slate construction / A/B:
  - `pipelines/slate/build_slate.py` (ranking, budget split heuristics)

Video feature extraction
- `pipelines/creative/extract_for_manifest.py` (reads `assets/` dirs) → `artifacts/creative/veo_*_features.parquet`
- Uses CLIP + YOLO (GPU if available)

Priors & bandits
- `pipelines/priors/*` → Beta priors and Thompson integration
- `pipelines/bandit/contextual_ts.py` → exploration policy for slates

Simulation + GA integration (later stage)
- `pipelines/generation/*`, `pipelines/landers/*`, `pipelines/ga/train_heads.py`
- `training_orchestrator/*` for sim→historical→real→scale progression

Operational
- `pipelines/deploy/package_bundle.py` (packaging), `pipelines/monitor/slice_report.py`

## 5) Models (what and where)

- CTR regressor (unique link CTR)
  - Path: `artifacts/models/ctr_creative_enhanced.joblib`
  - Target: `unique_inline_link_clicks / impressions` (KPI `ctr_u`)
  - Features: creative embeddings (CLIP/YOLO), text/image signals, temporal; pre-serve only (no spend leakage)
  - Use: ranking, expectation per impression; supports probability of ≥1 link click with `P = 1 - (1 - ctr_u)^N`

- Binary classifier (link>0)
  - Path: `artifacts/models/ctr_classifier.joblib`
  - Purpose: ranking/recall for sparse regimes; not a calibrated CTR

- Creative feature extractors
  - CLIP (text-image) and YOLO (object/scene) on GPU
  - Features parquet under `artifacts/creative/`

## 6) “First Day” Runbook (evaluate Meta ads, propose slate)

1) Login + env
- `gcloud compute ssh <you>@merlin-l4-1 --zone us-central1-c`
- `source /srv/aelp/.venv/bin/activate`
- `cd /home/harikravich_gmail_com/AELP`
- (auto env) or `source /srv/aelp/.env.local`

2) Refresh data and models (link-unique KPI)
- Fetch (read-only):
  ```
  PYTHONPATH=$PWD python3 tools/meta/fetch_ads_and_insights.py \
    --start 2025-09-01 --end 2025-10-01 --limit 500 --date-preset last_90d
  ```
- Unify + features + train:
  ```
  PYTHONPATH=$PWD python3 pipelines/data/unify_meta_marketing.py \
    --ads artifacts/meta/ads.csv \
    --ins artifacts/meta/insights.csv \
    --out artifacts/marketing/unified_ctr.parquet

  PYTHONPATH=$PWD python3 pipelines/features/join_creative.py \
    --unified artifacts/marketing/unified_ctr.parquet \
    --creative artifacts/creative/meta_creative_features.parquet \
    --out artifacts/features/marketing_ctr_joined.parquet

  PYTHONPATH=$PWD python3 pipelines/features/enhance_features.py \
    --in artifacts/features/marketing_ctr_joined.parquet \
    --train-end 2025-09-30 \
    --out artifacts/features/marketing_ctr_enhanced.parquet

  PYTHONPATH=$PWD python3 pipelines/ctr/train_ctr_creative.py \
    --data artifacts/features/marketing_ctr_enhanced.parquet \
    --out artifacts/models/ctr_creative_enhanced.joblib
  ```

3) Score current running + new creatives
- `PYTHONPATH=$PWD python3 pipelines/ctr/predict_ctr.py --model artifacts/models/ctr_creative_enhanced.joblib --data artifacts/features/marketing_ctr_latest_enhanced.parquet --out artifacts/predictions/ctr_scores_reg.parquet`
- Build daily “current running” table (script provided in session, or I can add a formal CLI)

4) Probability-of-success + slate
- For each ad: `P_click≥1(N) = 1 - (1 - pred_ctr_u)^N` with your daily impression target N
- Construct a slate with exploration weight (e.g., 80% exploit top performers, 20% explore new Balance concepts)
- Use `pipelines/slate/build_slate.py` as a starting point; I can expand to include uncertainty and frequency caps

## 7) Ad & Video Assets (where)

- Local (GPU VM)
  - Videos: `assets/veo_videos/*.mp4` (place new encodes here)
  - Thumbnails/creatives: `assets/...`
- Features and scores written to `artifacts/creative/` and `artifacts/predictions/`
- Remote (examples): `gs://gaelp-model-checkpoints-hariravichandran/coldstart/...`

Quick GCS inventory
- `GOOGLE_CLOUD_PROJECT=aura-thrive-platform ./scripts/checks/list_gcs_inventory.sh`
- Shows buckets and common prefixes; use `gsutil ls` / `gsutil du -s` for deeper dives

## 8) Architecture Diagram (high-level)

```mermaid
flowchart LR
  subgraph Sources
    A[Meta Ads API]\nads, insights
    V[Creative Assets]\nimages/videos
  end

  subgraph GPU_VM[GPU VM (merlin-l4-1)]
    F[tools/meta/fetch_ads_and_insights.py]
    U[pipelines/data/unify_meta_marketing.py]\n(ctr, link_ctr, ctr_u)
    C1[pipelines/creative/extract_*]\n(CLIP+YOLO)
    J[pipelines/features/join_creative.py]
    E[pipelines/features/enhance_features.py]
    T1[pipelines/ctr/train_ctr_creative.py]\n(CTR_u regressor)
    T2[pipelines/ctr/train_ctr_classifier.py]\n(link>0)
    P1[pipelines/ctr/predict_ctr.py]
    S[pipelines/slate/build_slate.py]
  end

  subgraph Storage
    GCS[(GCS buckets)]
    BQ[(BigQuery datasets)]
    ART[(Local artifacts/)]
  end

  subgraph Apps
    DASH[Dashboard (Cloud Run)]
  end

  A --> F --> U --> ART
  V --> C1 --> ART
  ART --> J --> E --> ART
  ART --> T1 --> ART
  ART --> T2 --> ART
  ART --> P1 --> ART
  ART --> S --> DASH
  ART --> BQ
  ART --> GCS
```

## 9) Security & Secrets

- All keys auto-load from `/srv/aelp/.env.local` for `aelp` group users (you)
- Secret Manager canonical copy: `AELP_DOTENV` (GCP)
- Please do not paste tokens in Slack or commit anything under `secrets/` or `*.env` to git

## 10) What’s Next / Ownership

- Move to conversion modeling: add GA4/Pixel conversions, compute expected subs = CTR_u × c2s; shift ranking/budgets by predicted conversions and CAC
- Stabilize reporting: 7‑day rolling CTR_u alongside today
- Expand simulation: generate, score, and slate concepts; track sim→real hit-rate
- Hardening: Optionally migrate artifacts to GCS, table-ize daily parquet into BigQuery, wire dashboard views

---
If you want this split into runbooks (daily/weekly) and a walkthrough video, I can add both.
