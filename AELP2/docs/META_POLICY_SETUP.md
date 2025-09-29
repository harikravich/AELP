# Meta Policy Setup — Exact Settings When A Creative Is Picked (Offline Playbook)

Use this as the checklist the team follows when promoting a simulator‑selected creative into a live Meta Sales campaign. It encodes the policy we assume in offline evaluation.

Campaign level
- Objective: Sales (web Purchase). Ensure the connected pixel is selected.
- Budgeting: 
  - Scale campaigns: CBO with a single broad ad set OR two sibling ad sets (Broad + LAL 1–5%).
  - Test cells: ABO (small budgets $100–$300/ad set/day) for incubation.
- AEM priorities: Purchase highest; add to the top of the AEM event list.
- Measurement hygiene: enable CAPI with event_id dedup; keep UTMs/naming convention consistent.

Ad set level (required policy for “compliant”)
- Optimization goal: `OFFSITE_CONVERSIONS` → event Purchase.
- Audience: Advantage Audience ON (aka detailed targeting expansion / Advantage audience).
- Placements: Advantage+ placements ON (auto placements). Do not prune unless a placement is ≥25% worse CAC over ≥7 days.
- Dynamic Creative (DCO): ON (or run a parallel DCO ad set). Load 6–10 distinct concepts per rotation; include 1:1 and 9:16 assets.
- Bid strategy:
  - Start with `LOWEST_COST_WITHOUT_CAP` until CAC stabilizes.
  - Then set cost cap ≈ 1.05–1.15 × trailing CAC and ratchet down if CAC remains under target for 3–5 days.
- Frequency: monitor 7‑day frequency; refresh at freq > 2.5 with rising CAC.
- Exclusions: exclude existing paying customers; keep remarketing separate from prospecting.

Ad level
- Creative formats: at minimum 1:1 (1080×1080) and 9:16 (1080×1920, Reels/Stories). Captions on; safe‑zones respected.
- Messaging: direct response for Purchase (e.g., “Get Protection in Minutes”); keep lead magnets in feeder campaigns only.
- Tracking: UTM source=meta, medium=paid_social, campaign/adset/ad name tokens.

Budget guardrails (align to offline policy)
- Test spend ≤ 10% of campaign/day while in incubation.
- Daily budget change ≤ 20% unless on scale track with stable CAC.
- Promote only when Purchase CAC ≤ 0.90 × control for ≥2 days and ≥40 purchases (or weekly dual‑gate passes).

Rollout steps when a creative is selected
1) Place selected creative into a DCO‑ON ad set that matches the above policy.
2) Start with ABO $100–$300/day (test cell) or add to Scale ASC if gates are already met.
3) Observe 3–5 days; if CAC ≤ target, raise budget +20%/day; otherwise recycle creative.
4) Once stabilized at target CAC, introduce cost cap slightly above trailing CAC and tighten.

Notes
- App ads (AEO/SKAN) should run in separate campaigns with distinct LTV goals and are not included in this Sales policy.
- All offline evaluation gates (dual CAC+volume, conformal bounds) assume these settings; deviations will degrade accuracy.

