#!/usr/bin/env python3
from __future__ import annotations
"""
Stub exporter for UI overlays: create timing JSON entries for proof inserts.
In practice, we will replace with real iOS/Android screen records.
Outputs: AELP2/competitive/ui_overlays.json
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'AELP2' / 'competitive' / 'ui_overlays.json'

def main():
    overlays = {
      'lock_card': { 'start_s': 4.8, 'end_s': 7.5, 'crop': [200, 600, 880, 1600], 'note': 'Card lock tap + toast' },
      'broker_removal': { 'start_s': 5.0, 'end_s': 8.0, 'crop': [120, 520, 960, 1700], 'note': 'Remove data broker' },
      'safe_browsing': { 'start_s': 5.0, 'end_s': 8.0, 'crop': [80, 480, 1000, 1720], 'note': 'Safe Browsing banner' }
    }
    OUT.write_text(json.dumps({'overlays': overlays}, indent=2))
    print(json.dumps({'overlays': list(overlays.keys())}, indent=2))

if __name__=='__main__':
    main()

