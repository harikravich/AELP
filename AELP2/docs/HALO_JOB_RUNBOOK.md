# Halo Job Runbook (GeoLift + Interference)

Prereqs
- R container available; permissions to write `halo_*` tables.

Steps (weekly)
1) Define test/control geos and channels in `halo_experiments`.
2) Run GeoLift container with spend/conv by geo; write `halo_reads_daily`.
3) Compute `channel_interference_scores` via regression from daily channel series.
4) Dashboard Ramp panel reads halo & interference and shows adjusted ROI.

Alerts
- If CI spans zero lift: mark as inconclusive; do not boost ramp.

