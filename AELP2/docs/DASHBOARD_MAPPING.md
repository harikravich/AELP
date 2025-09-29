# Dashboard Screen → API → Tables Mapping

Exec/Home
- KPIs: `/api/bq/kpi` → `ads_kpi_daily`
- Freshness: `/api/bq/freshness` → key tables
- Summary (LLM): `/api/chat` + BQ queries listed in proof

Growth Lab
- Bandit proposals: `/api/bq/offpolicy` → `bandit_change_proposals`
- Opportunities: `/api/bq/opportunities` → `platform_skeletons`
- Explore cells: `/api/bq/explore/cells` → `explore_cells`

RL Insights
- Policy: `/api/bq/rl/policy` → `bandit_posteriors`, `rl_policy_snapshots`

Creative Center
- Winners: `/api/bq/creatives` → `ads_ad_performance`/views
- Preview: `/api/ads/creative`
- Publish: `/api/control/creative/{enqueue,publish,rollback}` → `creative_publish_*`

LP Studio
- Tests list: `/api/bq/lp/tests` → `lp_tests`
- Publish: `/api/control/lp/publish`

Journeys & Halo
- GA4: `/api/bq/ga4/channels`, `/api/bq/journeys/sankey` → `ga4_*`
- Halo: `/api/bq/halo`, `/api/bq/interference` → `halo_*`

Experiments
- Definitions: `/api/bq/ab-experiments` → `ab_experiments`
- Exposures: `/api/bq/ab-exposures` → `ab_exposures`
- Results: `/api/bq/ab/results` → `ab_metrics_daily`

Control
- KPI Lock: `/api/control/kpi-lock`
- Ingest/Attribution: `/api/control/{ads-ingest,ga4-ingest,ga4-attribution}`
- Reach: `/api/control/reach-planner`
- Canary: `/api/control/{apply-canary,canary-rollback}`

Channels (new)
- Candidates list: `/api/research/channels` → `channel_candidates`
- Discover: `/api/research/discover` (Perplexity) → `channel_candidates`, `research_findings`
- Pilot CTA: creates 2–4 cells and a pilot record (via explore_cells + notes)
