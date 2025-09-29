# Halo & GeoLift Spec

Objective
- Measure brand‑lift and cross‑channel spillovers; correct ramp decisions.

Inputs
- Channel spend/conv daily by geo (state/region), brand search queries (if available), control/treatment share.

Process
- GeoLift (R): weekly runs, split geos; estimate lift and CI; write to `halo_reads_daily`.
- Interference scoring: simple regression of from_channel spend vs to_channel conv next day; write `channel_interference_scores`.

Outputs
- `halo_reads_daily(date, exp_id, brand_lift, ci_low, ci_high, method)`
- `channel_interference_scores(date, from_channel, to_channel, cannibalization, lift)`

Usage
- Volume Ramp panel: adjust expected ROI by (1 + brand_lift) and subtract cannibalization.

