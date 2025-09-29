# Migration Plan (AELP → AELP2)

Goals
- Reuse proven components (training_orchestrator, auction‑gym) for simulation/lab; keep prod loop in AELP2.

Steps
1) Catalog legacy modules and map to AELP2 equivalents (done in SYSTEM_INDEX.md).
2) Park lab‑only modules; export insights (not direct actions) to AELP2 via BQ hints.
3) Replace any prod‑path dashboards with Next.js app.
4) Migrate env/keys to Cloud Run secrets; standardize datasets.
5) Decommission redundant scripts; keep as lab references.

