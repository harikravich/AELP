# Research Agent (Perplexity) Spec

Goal
- Continuously discover new channels and creative/LP angles with citations, then produce structured, testable proposals.

Tools
- Perplexity API (Deep Research) with domain allow‑list and recency filters; no PII in prompts.

Env
- `PERPLEXITY_API_KEY` (service secret)

Tables
- `channel_candidates` (see SCHEMAS.sql.md)
- `research_findings` (see SCHEMAS.sql.md)

APIs
- `POST /api/research/discover { query, use_case? }` → writes candidates/findings
- `GET /api/research/channels?status=...` → lists candidates
- `POST /api/research/channels { name, type, ... }` → manual add

Prompts
- “List emerging ad networks suited to {use_case}. For each: audience fit, pricing, formats, targeting, min budget, API docs, policy risks, examples. Include 5–10 citations.”
- “For {use_case}, propose 10 angles with sources; include CPC/CVR drivers and policy notes.”

Safety & Logging
- Store citations; redact inputs; throttle queries; keep a daily budget limit.

Output → Action
- Candidates scored; those ≥ threshold auto‑marked “Recommend Pilot”, escalated to the Channels page for approval.

