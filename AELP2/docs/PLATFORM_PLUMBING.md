Platform Plumbing (Multi-Platform)

Objective
- Provide a platform-agnostic abstraction for actions (bid/budget/creative/targeting) and unified KPIs.
- Start with shadow-only stubs for Meta/TikTok/LinkedIn, with health checks.
- Keep orchestrator context/action schemas platform-aware via `platform_adapter`.

Adapters
- Base: `AELP2/core/data/platform_adapter.py` (NormalizedAction, UnifiedKPIs, PlatformAdapter)
- Google Ads: `AELP2/core/data/google_adapter.py` (shadow + partial live for bids/budgets)
- Meta: `AELP2/core/data/meta_adapter.py` (shadow-only)
- TikTok: `AELP2/core/data/tiktok_adapter.py` (shadow-only)
- LinkedIn: `AELP2/core/data/linkedin_adapter.py` (shadow-only)

Action Context
- Orchestrator builds a normalized action dictionary; subagents optionally include `platform` hints.
- Safety/HITL policies can be extended per platform (future) via env flags (e.g., `AELP2_POLICY_META_*`).

Next Steps
- Add platform selection logic in orchestrator/subagents.
- Define cross-platform KPI normalization and budget broker with per-platform spend guards.
- Add platform-specific HITL policy checks and rate limits.
