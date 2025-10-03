#!/usr/bin/env python3
"""
Optional live pilot scaffolding: Thompson Sampling explore with guardrails.

Runs an offline simulation with small traffic allocation and circuit breakers
for CPA and conversion rate.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from typing import Dict, List
import numpy as np


@dataclass
class Guardrails:
    min_cvr: float = 0.002
    max_cpa: float = 500.0


class TSPilot:
    def __init__(self, arms: Dict[str, Dict[str, float]], guardrails: Guardrails):
        self.arms = {k: {'alpha': v.get('alpha', 1.0), 'beta': v.get('beta', 1.0), 'spend': 0.0, 'convs': 0} for k, v in arms.items()}
        self.guardrails = guardrails
        self.paused = set()

    def sample(self, arm: str):
        a, b = self.arms[arm]['alpha'], self.arms[arm]['beta']
        return np.random.beta(a, b)

    def select_arm(self):
        candidates = [k for k in self.arms if k not in self.paused]
        if not candidates:
            return None
        return max(candidates, key=lambda k: self.sample(k))

    def update(self, arm: str, conversion: bool, cost: float):
        self.arms[arm]['spend'] += cost
        if conversion:
            self.arms[arm]['alpha'] += 1
            self.arms[arm]['convs'] += 1
        else:
            self.arms[arm]['beta'] += 1
        # Guardrails
        trials = self.arms[arm]['alpha'] + self.arms[arm]['beta'] - 2
        if trials >= 50:  # warmup
            cvr = self.arms[arm]['convs'] / max(1.0, trials)
            cpa = self.arms[arm]['spend'] / max(1, self.arms[arm]['convs'] or 1)
            if cvr < self.guardrails.min_cvr or cpa > self.guardrails.max_cpa:
                self.paused.add(arm)

    def run(self, steps=1000, cost_per_impr=0.01):
        rng = np.random.default_rng(0)
        for _ in range(steps):
            arm = self.select_arm()
            if arm is None:
                break
            p = self.sample(arm)
            conv = rng.random() < p
            self.update(arm, conv, cost=cost_per_impr)
        return {
            'paused': list(self.paused),
            'arms': self.arms,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=1000)
    args = ap.parse_args()
    pilot = TSPilot({'variant_a': {'alpha': 5, 'beta': 95}, 'variant_b': {'alpha': 10, 'beta': 90}}, Guardrails())
    res = pilot.run(steps=args.steps)
    print(res)


if __name__ == '__main__':
    main()

