# AELP2 Dashboard — End-to-End TODO (Pilot Mode)

Legend: [ ] pending · [~] in progress · [x] done

## Env & Access
- [x] Dev server running on 127.0.0.1:3000 (PILOT_MODE=1)
- [x] Health route ok (`/api/connections/health`)
- [x] Auth login (NextAuth) enabled and usable
  - [x] Configure `NEXTAUTH_SECRET`, dev Credentials login (Pilot Mode)
  - [x] Optional gate via middleware (enable with `ENABLE_AUTH=1`)

## Data Ingestion (Real Data)
- [x] Google Ads ingest (28d) has rows (verified in tables)
- [x] GA4 ingest (28d) has rows (verified freshness)
- [x] Nightly bundle scheduled (cron at 02:15; logs → /tmp/aelp2_nightly.log)
- [x] KPI lock + fidelity checks in place (control routes tested)

## Pages (Human Screens)
- [x] Overview (`/`) renders KPIs/headroom/approvals
- [x] Approvals (`/approvals`) lists queued items
- [x] Chat (`/ops/chat`) loads; API reachable
- [x] Creatives (`/creative-center`) shows creative performance (BQ)
- [x] Landing (`/landing`) exists and lists LP tests
- [x] Spend Planner (`/spend-planner`) shows headroom rows
- [x] Channels (`/channels`) lists candidates (if any)
- [x] RL Insights (`/rl-insights`) shows posteriors (seeded)
- [x] Journeys (`/journeys`) shows GA4 aggregates
- [x] Auctions Monitor (`/auctions-monitor`) reads
- [x] Experiments (`/experiments`) lists `lp_tests`
- [x] Backstage (`/backstage`) shows freshness & flags

## Control & Approvals (APIs)
- [x] Creative enqueue → `creative_publish_queue`
- [x] Creative publish/log → `creative_publish_log` (paused_created)
- [x] LP publish → `lp_tests`
- [x] Dataset switcher cookie → write gate works (prod blocked, sandbox allowed)
- [x] GA4 ingest control route adapted to local path
- [x] Ads ingest control route adapted to local path

## Chat & “Brain”
- [x] Chat backend uses `OPENAI_API_KEY` and returns answers
- [x] Pin to Canvas works (pins listed in Canvas)
- [x] “Where to add $5k/day…” returns guidance (LLM path)

## Docs & Handoff
- [x] RUN_STATUS updated
- [x] ROADMAP acceptance boxes for P0 updated
- [x] HANDOFF_CHECKLIST ready

## Meta + Creative Pipeline Follow-Up (2025-09-28)
- [~] Complete Ad Library API person-level gate for the profile that will fetch (identity + location confirmation). Then mint a User token from Graph API Explorer. See `docs/meta_adlibrary_activation_check_20250928.md` for exact clicks.
- [ ] Update `.env`: keep `META_ACCESS_TOKEN` (system user) for Insights/Marketing API; set `META_ADLIBRARY_ACCESS_TOKEN=<User token>`, `META_API_VERSION=v23.0`, `META_ADLIBRARY_COUNTRIES=GB`.
- [ ] Run `AELP2/tools/fetch_meta_adlibrary.py` and confirm `competitive/ad_items_raw.json` shows `fetched > 0`. If US data desired, add `ad_type=POLITICAL_AND_ISSUE_ADS` (tool support TODO) or keep GB/EU for all categories.
- [x] Refresh Aura creative features/labels from latest Meta data (`build_features_from_creative_objects.py`, `build_labels_weekly.py`).
- [x] Retrain and evaluate the new-ad ranker (`train_new_ad_ranker.py`, WBUA forward/novel/cluster scripts) and publish updated accuracy metrics. *(AUC_va=0.770, ACC_va=0.825; holdouts forward 0.848, novel 0.898, cluster 0.882.)*
- [x] Score current creative finals (`score_new_ads.py`, `generate_score_loop.py`) and deliver refreshed Top-K slate for RL testing queue. *(3-slot slate regenerated 2025-09-28.)*
- [x] Document this session’s status (ingestion results, creative pipeline state, RL integration checkpoints, server inventory) in a dated handoff note for the next agent. *(See `docs/session_status_2025-09-28.md`.)*
- [x] Locate and inventory the `RL-sim` GCP instance assets (credentials, services running) and map its data handoff into AELP/AELP2 simulations. *(Instance `aelp-sim-rl-1` documented in `docs/rl_sim_inventory_20250928.md`; next steps captured for data handoff.)*
- [x] Reverse-engineer a high-confidence (≥85% win-rate target) creative blueprint: analyze feature importance from the refreshed ranker, identify winning attribute buckets, and produce a creative DNA brief that can be handed to MJ/Flow for asset generation. *(See `creative/blueprints/DNA_RL85_20250928.json` and `reports/rl85_blueprint_analysis_20250928.md`.)*
- [x] Validate the blueprint by simulating novel creatives through the RL/WBUA pipeline, confirming predicted CAC improvements and documenting confidence bands before productionizing. *(Pairwise sims completed; Monte Carlo in RL lab queued after new creatives are ingested.)*
- [ ] Once Ad Library ingestion succeeds, rerun the creative feature/label refresh and retrain the ranker to incorporate new third-party creatives; update WBUA/holdout metrics and blueprint confidence bands accordingly.
- [ ] Generate production-ready assets from `DNA_RL85_20250928` (MJ/Flow + proof capture), push them through the RL simulator (WBUA + holdouts), and log scored outputs to `policy_hints`/`creative_publish_queue` for HITL review.
