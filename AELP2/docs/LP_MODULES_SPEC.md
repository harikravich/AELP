# Landing Page Modules (Proof Blocks) — Engineering Spec

Goal
- Add small, impressive, safe “proof” blocks to LPs that lift trust and CVR.

Modules (initial)
- insight_preview (Balance): public/allowed social signals → histogram/anomaly → plain text.
- scam_check (Fraud): URL features → risk score (rules + tiny model) → traffic light + “why” bullets.
- privacy_check (Demo): seeded breach summary (live partner later, post-legal).

Flow
1) LP posts to `/api/module/:slug/start` with consent and input; returns `run_id`.
2) Background job runs connector/model; writes `module_results`.
3) LP polls `/status` then `/result` to render summary and tiny chart.

Data & Tables
- `lp_module_runs(run_id, slug, page_url, consent_id, created_ts, status, elapsed_ms, error_code)`
- `consent_logs(consent_id, slug, page_url, consent_text, ip_hash, user_agent, ts)`
- `module_results(run_id, slug, summary_text, result_json, expires_at)`

Safety
- Consent required; no raw PII stored; hash inputs where possible; short retention.
- Flags: `AELP2_LP_MODULES_ENABLED`, `AELP2_MODULE_<SLUG>_LIVE` (0 default).

Perf Budgets
- enqueue <200ms; job ≤1.5s typical; summarizer ≤300ms; async progress UI.

LLM Usage
- Only to rephrase sanitized signals; never fetch protected content; “Show Proof” reveals raw indicators.

Code Layout
- Next.js: `src/app/api/module/[slug]/{start,status,result}.ts` ; `src/components/lp/modules/*`.
- Python: `AELP2/pipelines/module_runner.py`; `connectors/social_signals_lite.py`, `scam_link_risk.py`, `breach_demo.py`.

