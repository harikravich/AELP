#!/usr/bin/env python3
"""
Integrate Beta priors into Thompson Sampling strategies (offline test).

Usage:
  python pipelines/priors/integrate_priors_thompson.py --priors artifacts/priors.json --out artifacts/ts_strategies.json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict

from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from production_online_learner import ThompsonSamplingStrategy


def build_strategies(priors: Dict) -> Dict[str, Dict]:
    strategies = {}
    for key, metrics in priors.items():
        # Use c2s priors for conversion probability
        a = float(metrics['c2s']['alpha'])
        b = float(metrics['c2s']['beta'])
        ts = ThompsonSamplingStrategy(strategy_id=str(key), prior_alpha=a, prior_beta=b)
        # Run a few offline samples to verify behavior
        _ = [ts.sample_probability() for _ in range(5)]
        strategies[str(key)] = {
            'alpha': ts.alpha,
            'beta': ts.beta,
            'expected': ts.get_expected_value(),
            'ci_95': ts.get_confidence_interval(0.95),
        }
    return strategies


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--priors', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    priors = json.loads(Path(args.priors).read_text())
    strategies = build_strategies(priors)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(strategies, indent=2))
    print(f"Wrote Thompson strategies with priors to {out}")


if __name__ == '__main__':
    main()
