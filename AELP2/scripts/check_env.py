#!/usr/bin/env python3
import os
import sys

REQUIRED = [
    'GOOGLE_CLOUD_PROJECT',
    'BIGQUERY_TRAINING_DATASET',
    'AELP2_MIN_WIN_RATE',
    'AELP2_MAX_CAC',
    'AELP2_MIN_ROAS',
    'AELP2_MAX_SPEND_VELOCITY',
    'AELP2_APPROVAL_TIMEOUT',
    'AELP2_SIM_BUDGET',
    'AELP2_SIM_STEPS',
    'AELP2_EPISODES',
]

def main() -> int:
    missing = [v for v in REQUIRED if not os.getenv(v)]
    if missing:
        print(f"Missing required env vars: {', '.join(missing)}")
        return 1
    ref = os.getenv('AELP2_CALIBRATION_REF_JSON')
    if ref:
        ks = os.getenv('AELP2_CALIBRATION_MAX_KS')
        mse = os.getenv('AELP2_CALIBRATION_MAX_HIST_MSE')
        if not ks or not mse:
            print("AELP2_CALIBRATION_MAX_KS and AELP2_CALIBRATION_MAX_HIST_MSE are required when AELP2_CALIBRATION_REF_JSON is set")
            return 2
    print("Environment configuration looks good.")
    return 0

if __name__ == '__main__':
    sys.exit(main())

