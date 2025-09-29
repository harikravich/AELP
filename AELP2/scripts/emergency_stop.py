#!/usr/bin/env python3
"""
Trigger an emergency stop safety event with a reason and optional context.

Usage:
  python -m AELP2.scripts.emergency_stop --reason "manual_stop" --note "Nightly drill"

Env:
  Must have GOOGLE_CLOUD_PROJECT and BIGQUERY_TRAINING_DATASET set to enable BQ logging via writer.
"""

import argparse
import os
from datetime import datetime

from AELP2.core.safety.hitl import emergency_stop
from AELP2.core.monitoring.bq_writer import create_bigquery_writer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reason", required=True)
    p.add_argument("--note", required=False)
    args = p.parse_args()

    ctx = {
        'invoked_at': datetime.utcnow().isoformat(),
        'note': args.note,
    }
    # Log to BQ as well for audit trail
    try:
        bq = create_bigquery_writer()
        bq.write_safety_event({
            'event_type': 'emergency_stop',
            'severity': 'critical',
            'metadata': {'reason': args.reason, 'context': ctx}
        })
    except Exception:
        pass
    emergency_stop(args.reason, context=ctx)


if __name__ == "__main__":
    main()

