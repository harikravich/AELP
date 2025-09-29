# Landing Page Studio Spec

Blocks
- Hero (headline, sub, image/video), Benefits, Trust (badges), FAQ, Pricing Anchor, Sticky CTA, Social Proof
- Dynamic module (optional): consent-gated “insight preview” (shadow by default)

Data
- `lp_tests(test_id, lp_a, lp_b, status, traffic_split, primary_metric)`
- `lp_block_metrics(date, lp_url, block, metric, value)`
- `funnel_dropoffs(date, lp_url, stage, visitors, drop_rate)`

APIs
- `POST /api/control/lp/publish` (create/update tests; default 50/50)
- `GET /api/bq/lp/tests` (list)

Routing
- Assignment via `/api/ab/assign` cookie; LP reads cookie to choose A/B; GA4 events capture funnels.

Code Architecture (modules)
- Registry: `slug`, input schema, consent text, cooldown, handlers.
- Endpoints: `/api/module/:slug/start|status|result` (server-side only; no secrets on client).
- Runner: Cloud Run job executes connectors/models; writes sanitized `module_results`.
- Connectors: `social_signals_lite`, `scam_link_risk`, `breach_demo` (P0); partner APIs later.
- Safety: consent required; flags `AELP2_LP_MODULES_ENABLED=1`, `AELP2_MODULE_<SLUG>_LIVE=0` default.
