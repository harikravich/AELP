Data Sources Mapping (External UI → Next.js APIs → BigQuery)

Global
- Base URL: `VITE_API_BASE_URL` (e.g., same-origin or Next.js domain). All requests `credentials: 'include'` to use dataset cookie via `/api/dataset`.
- Dataset: `GET /api/dataset` (mode,dataset), `POST /api/dataset?mode=sandbox|prod` to switch.

ExecutiveDashboard
- KPI tiles: `GET /api/bq/kpi` and `GET /api/bq/kpi/daily?days=28` → table/view `ads_kpi_daily`.
- Headroom: `GET /api/bq/headroom` → `mmm_allocations` fallback heuristic.
- Freshness badge: `GET /api/bq/freshness` → per-table max(date).
- Channel mix/device perf: derive from `GET /api/bq/cross-kpi` + `GET /api/bq/ga4/channels`.

CreativeCenter
- Top ads by performance: `GET /api/bq/creatives` → `ads_ad_performance`.
- Copy suggestions/variants: `GET /api/bq/copy-suggestions`, `GET /api/bq/creative-variants`.
- Live preview: `POST /api/ads/creative` (Google Ads v19 preview) with `{customerId, adId}`.

SpendPlanner
- Headroom cards: `GET /api/bq/headroom`, `GET /api/bq/mmm/allocations`, `GET /api/bq/mmm/curves?channel=...`.
- Apply plan (paused/HITL): `POST /api/control/bandit-apply` or queue via Approvals.

Approvals
- Queue list: `GET /api/bq/opportunities` & BQ query in `creative_publish_queue`.
- Approve/apply: `POST /api/control/opportunity-approve`, `POST /api/control/apply-creative`, `POST /api/control/apply-canary`.

Finance
- Trends/ROAS/CAC: `GET /api/bq/kpi/daily?days=90`, `GET /api/bq/cross-kpi`.
- Fidelity: `GET /api/bq/fidelity`.

RLInsights
- Bandit: `GET /api/bq/offpolicy`, `GET /api/bq/interference`, `GET /api/bq/subagents`.

TrainingCenter
- Status/health: `GET /api/control/status`, `GET /api/ops/flows`.
- Trigger run: `POST /api/control/training-run`.

OpsChat
- Chat w/ context: `POST /api/chat` with prompt; server injects KPI + recent rows.

Channels
- Attribution/mix: `GET /api/bq/channel-attribution`, `GET /api/bq/ga4/channels`, `GET /api/bq/mmm/channels`.

Experiments
- AB tests: `GET /api/bq/ab-experiments`, exposures: `GET /api/bq/ab-exposures`.
- LP tests: `GET /api/bq/lp/tests` and `GET /api/bq/lp-ab`.
- Approve/publish: `POST /api/control/ab-approve`.

AuctionsMonitor
- Realtime: `GET /api/bq/auctions/minutely`.
- Policy: `GET /api/bq/policy-enforcement`, Ops alerts: `GET /api/bq/ops-alerts`.
- Bidding: `GET /api/bq/bid-landscape`, edits: `GET /api/bq/bid-edits`.

LandingPages
- Builder modules: `POST /api/module/[slug]/start|status|result`.
- Publish LPs: `POST /api/control/lp/publish` (paused/HITL).

