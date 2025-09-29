#!/usr/bin/env python3
from __future__ import annotations
"""
Finalize a candidate video with hard gates so low-quality outputs never publish.

Checks:
- QC gates (aspect/text/blur/loudness) via qc_gates.py --role final
- Self-judge scores via self_judge.py (Interestingness, Relevance)

Thresholds (defaults, override via CLI):
- Relevance >= 0.30
- Interestingness >= 0.60

Behavior:
- If pass: write <stem>_gate.json with pass=true and leave file in finals.
- If fail: move MP4 + poster (if exists) to AELP2/outputs/rejected and write reasons.

Usage:
  python3 AELP2/tools/finalize_gate.py --video AELP2/outputs/finals/foo.mp4 \
      --thr-rel 0.30 --thr-int 0.60
"""
import json, shutil, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FINALS = ROOT / 'AELP2' / 'outputs' / 'finals'
REJECT = ROOT / 'AELP2' / 'outputs' / 'rejected'

def run_json(cmd: list[str]) -> dict:
    try:
        out = subprocess.check_output(cmd).decode()
        return json.loads(out)
    except Exception as e:
        return {'error': str(e), 'raw': None}

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--thr-rel', type=float, default=0.30)
    ap.add_argument('--thr-int', type=float, default=0.60)
    args = ap.parse_args()

    vid = Path(args.video)
    assert vid.exists(), f"missing {vid}"

    qc = run_json(['python3', str(ROOT/'AELP2'/'tools'/'qc_gates.py'), '--video', str(vid), '--role', 'final'])
    jd = run_json(['python3', str(ROOT/'AELP2'/'tools'/'self_judge.py'), '--video', str(vid)])

    interesting = (jd.get('scores') or {}).get('interestingness')
    relevance = (jd.get('scores') or {}).get('relevance')
    qc_pass = bool(qc.get('pass'))
    pass_rel = (relevance is not None) and (relevance >= args.thr_rel)
    pass_int = (interesting is not None) and (interesting >= args.thr_int)
    passed = qc_pass and pass_rel and pass_int

    gate = {
        'video': str(vid),
        'thresholds': {'relevance': args.thr_rel, 'interestingness': args.thr_int},
        'qc': qc,
        'judge': jd,
        'pass': passed,
        'reasons': []
    }
    if not qc_pass:
        gate['reasons'].append('qc_failed')
    if not pass_rel:
        gate['reasons'].append('relevance_below_threshold')
    if not pass_int:
        gate['reasons'].append('interestingness_below_threshold')

    # Write gate json next to the video (even on failure)
    gate_path = vid.with_name(vid.stem + '_gate.json')
    gate_path.write_text(json.dumps(gate, indent=2))

    if not passed:
        REJECT.mkdir(parents=True, exist_ok=True)
        dst = REJECT / vid.name
        # Move MP4 and its poster if present
        poster = vid.with_suffix('.jpg')
        shutil.move(str(vid), str(dst))
        if poster.exists():
            shutil.move(str(poster), str(REJECT/poster.name))
        print(json.dumps({'status': 'rejected', 'moved_to': str(dst), 'gate_json': str(gate_path)}, indent=2))
    else:
        print(json.dumps({'status': 'approved', 'video': str(vid), 'gate_json': str(gate_path)}, indent=2))

if __name__ == '__main__':
    main()

