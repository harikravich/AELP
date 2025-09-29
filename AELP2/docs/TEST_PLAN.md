# QA / Test Plan

Functional
- Serializer everywhere; error cards render; freshness shows; dataset switch works.
- Experiments: assign/expose/log; results populate; SRM check flags.
- Explore cells: add/list; bandit posteriors generated (seed or real).
- Creative publisher: enqueue→publish (DRY)→log; rollback no‑ops.
- LP Studio: publish two variants; GA4 funnels captured; lp_tests rows exist.
- Halo job: writes halo_reads_daily; ramp drawer shows halo‑adjusted ROI.

Safety
- Writes blocked on prod by default; flags required for live actions; all actions logged.
- Per‑change and daily caps enforced; rollback works.

Performance
- All dashboard pages render <2s P50 server time on cached queries or simple views.

