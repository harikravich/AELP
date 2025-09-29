# Meta Balance — Auction vs Thompson (Test Plan)

**TL;DR**
- Auction-aware slate looks better than Thompson on our calibrated forecast: ~10% lower CAC and ~13% more signups at $30k/day.
- Top‑8 to set up now (AU): `bp_0116, bp_0001, bp_0113, bp_0114, bp_0041, bp_0006, bp_0003, bp_0005`.
- Guardrails: caps, early‑stops, 70/30 split (AU/TS) for 72h, with daily check against p10–p90 bounds.

**What changed**
- We ran an auction‑style simulator (AuctionGym) with CPM calibration to our baselines and compared slates.

**Results (p50 @ $30k/day)**
- TS slate: avg CAC $255.06; sum signups 965.3.
- AU slate: avg CAC $228.55; sum signups 1091.3.
- Stability: AU top‑5 consistent under bid ±20% and quality ±10%.

**Setup**
- Campaign: Balance — US — FB/IG — $30k/day.
- Split: 70% AU slate / 30% TS slate (8 ads each).
- Caps: per‑ad 20% of daily; auto‑pause if CAC > p90 for 2 consecutive days or <50 signups/day.
- Early‑stop: stop a creative if its CAC > p90 and signups < p10 for 2 consecutive days; backfill with next from AU list.

**QA**
- Creative IDs and copy/asset checks in planner.
- Policy pre‑flight for teen/parent wellbeing claims.

**Links**
- AU report: `AELP2/reports/auctiongym_offline_simulation_calibrated.json`
- TS ranking: `AELP2/reports/rl_offline_simulation.json`
- Holdout eval: `AELP2/reports/holdout_evaluation.json`, `.../holdout_significance.json`

