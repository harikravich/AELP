# API Contracts (New/Extended)

Base path: Next.js App Router `/api/*`. All server-side, no client secrets.

## Experiments & Assignments
- `POST /api/ab/assign`
  - Body: `{ experiment: string, unit_id: string, unit_type?: 'ga4_client_id'|'user_id'|'cookie'|'ad_click_id', context?: object }`
  - Returns: `{ variant: string, assigned: boolean }`
  - Side‑effect: upsert `${DATASET}.ab_assignments` row.
- `POST /api/ab/exposure` (exists, extend validation)
  - Body: `{ experiment: string, variant: string, subject_id: string, context?: object }`
  - Returns: `{ ok: true }`
- `GET /api/bq/ab/results`
  - Returns: `{ rows: Array<{ date, experiment, variant, spend, clicks, conversions, revenue, cac, roas }> }`

## Explore & RL
- `GET /api/bq/explore/cells`
  - Returns: `{ rows: Array<{ cell_key, angle, audience, channel, lp, offer, last_seen, spend, conversions, cac, value }> }`
- `POST /api/bq/explore/cells`
  - Body: `{ angle, audience, channel, lp, offer }`
  - Returns: `{ ok: true, cell_key }`
- `GET /api/bq/rl/policy`
  - Returns: `{ rows: Array<{ cell_key, metric, mean, ci_low, ci_high, samples }> }`

## Creative Publisher
- `POST /api/control/creative/enqueue`
  - Body: `{ platform: 'google_ads'|'meta'|'tiktok', type: 'rsa'|'pmax_asset_group'|'video', campaign_id?: string, ad_group_id?: string, asset_group_id?: string, payload: object }`
  - Returns: `{ ok: true, run_id }`
- `POST /api/control/creative/publish`
  - Body: `{ run_id?: string }` (if omitted, process oldest queued)
  - Returns: `{ ok: true, summary: { created: number, errors: number } }`
- `POST /api/control/creative/rollback`
  - Body: `{ platform: string, platform_ids: object }`
  - Returns: `{ ok: true }`

## Research (Perplexity)
- `POST /api/research/angles`
  - Body: `{ use_case: string, max?: number, recency_days?: number }`
  - Returns: `{ rows: Array<{ angle, audience, channel, lp, offer, rationale, expected_cac_min, expected_cac_max, sources }> }`
- `POST /api/research/brief`
  - Body: `{ use_case?: string, angle: string }`
  - Returns: `{ brief: { hook, copy_guidelines, policy_flags, examples, sources } }`

## Landing Pages
- `POST /api/control/lp/publish`
  - Body: `{ test_id?: string, lp_a: string, lp_b?: string, traffic_split?: number, primary_metric?: string }`
  - Returns: `{ ok: true, test_id }`
- `GET /api/bq/lp/tests`
  - Returns: `{ rows: Array<{ test_id, created_at, lp_a, lp_b, status, traffic_split, primary_metric }> }`

## Landing Page Modules (Proof Blocks)
- `POST /api/module/:slug/start`
  - Body: `{ input: object, consent: boolean, page_url: string }`
  - Returns: `{ ok: true, run_id, status: 'queued'|'running' }`
- `GET /api/module/:slug/status?run_id=...`
  - Returns: `{ status: 'queued'|'running'|'done'|'error', elapsed_ms?, error_code? }`
- `GET /api/module/:slug/result?run_id=...`
  - Returns: `{ ok: true, summary_text, result_json }`
Notes: server validates consent, rate‑limits, and never stores raw PII. Results expire.

## Halo & Interference
- `GET /api/bq/halo`
  - Returns: `{ rows: Array<{ date, exp_id, brand_lift, ci_low, ci_high, method }> }`
- `GET /api/bq/interference`
  - Returns: `{ rows: Array<{ date, from_channel, to_channel, cannibalization, lift }> }`

## Security & Safety
- All control routes enforce:
  - Mode: `resolveDatasetForAction('write')` must allow (no writes on prod by default)
  - Flags: `GATES_ENABLED=1`, action‑specific `ALLOW_*=1`
  - Audit: insert into `${DATASET}.ops_flow_runs` / relevant logs

## Research & Channels (new)
- `GET /api/research/channels`
  - Query: `status?=new|triage|pilot|live|archived`
  - Returns: `{ rows: Array<channel_candidate> }`
- `POST /api/research/channels`
  - Body: `{ name, type, use_cases[], docs_url?, notes? }` (manual add)
  - Returns: `{ ok: true, id }`
- `POST /api/research/discover`
  - Body: `{ query: string, use_case?: string }`
  - Effect: calls Perplexity with allow‑listed domains; writes `channel_candidates` + `research_findings`
  - Returns: `{ ok: true, created: number }`
