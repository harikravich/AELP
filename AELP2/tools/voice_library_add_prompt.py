#!/usr/bin/env python3
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LIB = ROOT / 'AELP2' / 'branding' / 'voice_library.json'

def main():
    prompt = sys.argv[1] if len(sys.argv)>1 else None
    name = sys.argv[2] if len(sys.argv)>2 else None
    if not prompt or not name:
        print('usage: voice_library_add_prompt.py "<prompt>" "<name>"')
        return
    lib={'voices': []}
    if LIB.exists():
        try: lib=json.loads(LIB.read_text())
        except Exception: pass
    lib['voices'].append({'name': name, 'prompt': prompt, 'voice_id': None, 'status': 'prompt_only'})
    LIB.write_text(json.dumps(lib, indent=2))
    print(json.dumps({'added': name}, indent=2))

if __name__=='__main__':
    main()

