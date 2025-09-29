# Rehydrate Instructions (Next Codex Session)

This repo already has a machine-readable state file at `AELP2/rehydrate_state.json`. Load that to restore context (servers, gates, reports, assets, resume steps).

## Quick Resume
- Finals viewer: run `python3 -m http.server 8080 --bind 127.0.0.1 --directory AELP2/outputs/finals` then open http://localhost:8080/
- Rejected viewer (review only): `python3 -m http.server 8081 --bind 127.0.0.1 --directory AELP2/outputs/rejected` → http://localhost:8081/
- Hard gate thresholds: Relevance ≥ 0.30; Interestingness ≥ 0.60 (see `AELP2/tools/finalize_gate.py`).

## Accuracy (Aura only)
- Forward WBUA ≈ 0.848 — `AELP2/reports/wbua_forward.json`
- Novel WBUA ≈ 0.898 — `AELP2/reports/wbua_novel_summary.json`
- Cluster WBUA ≈ 0.882 — `AELP2/reports/wbua_cluster_summary.json`

## Scorer
- Model: `AELP2/models/new_ad_ranker/` (AUC_va≈0.723, Acc_va≈0.810).
- Score candidates: `python3 AELP2/tools/score_new_ads.py`
- Pick slate: `python3 AELP2/tools/generate_score_loop.py`

## Create 3 “Boringly Good” Variants
1) Put 3 upscaled MJ singles in `AELP2/assets/backplates/mj/imported/`.
2) Put a 3–5s real screen recording (lock card or breach results) in `AELP2/assets/proof_raw/`.
3) Ingest & animate: `python3 AELP2/tools/mj_ingest.py --src AELP2/assets/backplates/mj/imported --animate`
4) Import proof: `python3 AELP2/tools/import_proof_clips.py`
5) Assemble: `python3 AELP2/tools/assemble_boring_good.py` (auto-gates and publishes to viewers).

## Files of Interest
- State: `AELP2/rehydrate_state.json`
- Reports: `AELP2/reports/*`
- Assets: `AELP2/assets/*`
- Tools: `AELP2/tools/*`

---
Generated this session. Update if gates or ports change.
