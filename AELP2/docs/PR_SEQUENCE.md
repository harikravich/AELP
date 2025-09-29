# PR Sequence & Ownership (P0 → P1)

P0 PRs
1) FE/BE: Serializer enforcement + error cards + freshness badges
   - Touch: `apps/dashboard/src/app/**` pages using raw BigQuery; wrap with `createBigQueryClient`
   - Accept: no RSC/Invalid Date; friendly error cards
2) Platform: A/B assignments + metrics thin slice
   - Add: tables `ab_assignments`, `ab_metrics_daily` (SCHEMAS.sql.md)
   - Add: `/api/ab/assign`, extend `/api/ab/exposure`, `/api/bq/ab/results`
   - Page: Experiments v1
3) Growth Eng: Creative publisher scaffolding
   - Add: tables `creative_publish_queue`, `creative_publish_log`
   - Add: `/api/control/creative/{enqueue,publish,rollback}` (publish queued only, DRY + paused)
   - Script: `pipelines/publish_google_creatives.py` (RSA/PMax)
4) FE/BE: Explore cells scaffolding
   - Add: table `explore_cells` + `/api/bq/explore/cells` (GET/POST)
   - Page: Growth Lab list and manual add
5) Research Agent (Perplexity) scaffolding
   - Add: env `PERPLEXITY_API_KEY`; spec doc; allow‑list domains
   - Add: tables `channel_candidates`, `research_findings`
   - API: `/api/research/channels` (GET/POST), `/api/research/discover` (Perplexity call)
6) LP Modules (framework + demo blocks)
   - Add: tables `lp_module_runs`, `consent_logs`, `module_results`
   - API: `/api/module/:slug/{start,status,result}`; registry; demo Insight + Scam blocks
   - A/B: turn on /balance‑insight‑v1 vs /balance‑v2

P1 PRs
5) DS: Bandits/MABWiser integration + posteriors persistence
   - Update: `core/optimization/bandit_service.py`; write `bandit_posteriors`, `rl_policy_snapshots`
   - Page: RL Insights
6) DS: MMM headroom APIs + Volume Ramp page
   - Add: `/api/bq/mmm/{curves,allocations}`; UI sliders + HITL request
7) DS: Halo v1 (GeoLift weekly + interference)
   - Add tables `halo_experiments`, `halo_reads_daily`, `channel_interference_scores`; runbook
8) Growth Eng: Creative Center v1
   - Winners view; policy lint; 3 variants queued; preview
9) Growth Eng: LP Studio v1
   - Two templates; A/B routing via assignments; GA4 funnels; lp_tests logging
10) Platform: Audience exports (shadow)
   - Queue exports; write logs; UI controls in Onboarding
11) FE/BE: Channels page (Candidates → Pilots → Live)
   - Read `channel_candidates`; pilot CTA; show proofs and scores
12) LP Modules live‑lite
   - Insight Preview (public/allowed live signals), Scam Check v2 (tiny model)
   - Privacy Check (demo) behind legal flag

11) Platform: Research Agent (Perplexity)
   - Tables: `research_angle_candidates`, `research_findings`, `creative_briefs`
   - APIs: `/api/research/angles`, `/api/research/brief`
   - Growth Lab: “Generate ideas” button; results into Explore Cells

Tracking
- Each PR links acceptance in ROADMAP.md. Update RUN_STATUS.md after execution.
