#!/usr/bin/env python3
import sys
import subprocess


def test_gx_dry_run_pass():
    rc = subprocess.run([sys.executable, 'AELP2/ops/gx/run_checks.py', '--dry_run'], check=False).returncode
    assert rc == 0


def test_gx_dry_run_fail_injected():
    rc = subprocess.run([sys.executable, 'AELP2/ops/gx/run_checks.py', '--dry_run', '--inject_bad'], check=False).returncode
    assert rc == 1


if __name__ == '__main__':
    test_gx_dry_run_pass(); test_gx_dry_run_fail_injected(); print('gx_checks tests OK')

