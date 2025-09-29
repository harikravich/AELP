# Audience Sync Spec (Shadow → Live)

Scope
- Export top segments/lookalikes to ad platforms with caps and audit.

Data
- `segment_scores_daily(segment, date, score)`
- export logs in users dataset when live

Flow
1) Select segments (threshold/size); 2) Queue export (shadow); 3) On approval, push to platform via adapters; 4) Audit IDs.

Safety
- Opt‑out honored; hashed identifiers; privacy review before live.

