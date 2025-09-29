Demo Script (External Dashboard wired to AELP2)

Pre-req
- Next.js API running at http://localhost:3000 with Sandbox dataset.
- This app running with VITE_API_BASE_URL=http://localhost:3000.

Flow (5–7 minutes)
1) Executive: show live CAC/ROAS tiles, headroom list; point out dataset mode in header.
2) Creative Center: Top Ads by LP; pick a top ad, open Live Preview (Ad final URL); click "Scale Winner" → shows queued in Approvals.
3) Approvals: see queued creative; click Approve → apply via control API (respects flags).
4) Spend Planner: review headroom suggestions; click "Queue Plan" to log approval; optionally trigger Bandit Apply (shadow).
5) Channels: skim channel CAC/ROAS and daily spend.
6) RL Insights: show recent AI decisions (offpolicy log).
7) Training Center: show ops status/events.
8) Finance: check CAC/ROAS tiles; run Meta/Google value upload triggers.

Notes
- All write actions are flag-gated server-side and run in Pilot mode by default.
- The dataset cookie is set on first load based on `VITE_DATASET_MODE`.
