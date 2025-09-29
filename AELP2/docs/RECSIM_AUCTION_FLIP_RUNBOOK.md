# Flip to Auction/RecSim — Offline Insights (Runbook)

Goal: run an auction- and user-simulation–aware offline study side‑by‑side with the current Thompson Sampling pipeline to see if we learn deeper insights (bid dynamics, fatigue, persona sensitivity) before changing production.

## Why we aren’t using RecSim/Auction today
- Production is optimized around simple, robust Thompson Sampling with calibrated placement baselines and has delivered fast, explainable results.
- RecSim NG + AuctionGym require heavy, brittle stacks (TF/TFP/Edward2/JAX, PyTorch/Numba). On this machine we hit version conflicts (e.g., Edward2 with SciPy/Numpy; AuctionGym with NumPy/Numba; Torch needing newer `typing_extensions`).

## What we will run (today)
1) Auction‑aware offline sim (no RecSim): `AELP2/tools/simulate_auctiongym_from_baselines.py`
   - Uses current forecasts to derive CTR/CVR per creative, then runs a second‑price auction via AuctionGym to estimate win rate, CPC, CPM, CAC under competition.
   - Output: `AELP2/reports/auctiongym_offline_simulation.json`.

2) (Optional) Full RecSim pass (personas + journeys) — requires isolated environment.

## Recommended environment (virtualenv)

```bash
cd ~/AELP
python3 -m venv .venv_recsim
source .venv_recsim/bin/activate
python -m pip install --upgrade pip

# Minimal for AuctionGym
pip install numpy==1.24.3 numba==0.58.1 typing_extensions>=4.12 torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu

# Optional: RecSim NG stack (may take time and can be finicky)
pip install tensorflow==2.13.1 tensorflow-probability==0.21.0 dm-tree==0.1.8 recsim-ng==0.2.0 edward2
```

If Torch/Numba conflict with TF versions, prefer a separate venv just for AuctionGym.

## Run AuctionGym offline sim

```bash
source .venv_recsim/bin/activate
python AELP2/tools/simulate_auctiongym_from_baselines.py
cat AELP2/reports/auctiongym_offline_simulation.json | head -n 60
```

## (Optional) RecSim demo

```bash
source .venv_recsim/bin/activate
python install_recsim.py
python run_recsim_demo_fixed.py
```

## Compare vs Thompson Sampling

Artifacts to compare:
- Bandit: `AELP2/reports/rl_offline_simulation.json` (current TS)
- Auction: `AELP2/reports/auctiongym_offline_simulation.json`

Key checks:
- CAC deltas vs forecast (`delta_cac_vs_forecast` by creative)
- Rank shifts in top‑8 slate
- Win‑rate sensitivity to bid/quality (qualitative)

## Next steps if RecSim stabilizes
- Calibrate RecSim personas to placement baselines (map CTR/CVR, fatigue, time‑of‑day)
- Extend planner to read `recsim_offline_simulation.json` and show a “simulation‑aware” rank alongside TS.
- Keep production on TS until RecSim adds measurable lift (e.g., better early‑stop/portfolio rotation decisions).

