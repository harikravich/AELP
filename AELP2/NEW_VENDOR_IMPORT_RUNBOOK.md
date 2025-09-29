New Vendor Creative Import — Quick Runbook

Goal: Use third‑party Meta ad exports (BigSpy/PowerAdSpy) to score creatives with our trained new‑ad ranker while Ad Library access is pending.

What you do
- Export last 30–90 days of Facebook/Instagram ads to CSV (preferably English, US+GB; include headline/title, text, platform(s), media type, CTA, destination URL, page ID/name, first_seen/last_seen if available).
- Drop the file at `AELP2/vendor_imports/` with a helpful name (e.g., `bigspy_20250928_meta.csv`).

What to run
  source .env
  python3 AELP2/tools/import_vendor_meta_creatives.py --src AELP2/vendor_imports
  python3 AELP2/tools/build_features_from_creative_objects.py
  python3 AELP2/tools/score_vendor_creatives.py

Outputs
- Normalized creative objects: `AELP2/reports/creative_objects/vendor_*.json`
- Feature file: `AELP2/reports/creative_features/creative_features.jsonl`
- Scores: `AELP2/reports/vendor_scores.json` (sorted by predicted win prob)

Notes
- We don’t retrain the model on vendor data (no reliable labels). We score them to shortlist candidates for RL/WBUA and production testing.
- Once Ad Library access is live, we’ll ingest GB/EU creatives with real metadata and optionally retrain to improve coverage/generalization.

